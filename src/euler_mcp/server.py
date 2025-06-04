"""Main MCP Server for EulerMCP."""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from mcp import Tool
from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import BaseModel

from .conversation_processor import ConversationProcessor
from .graph_engine import LearningGraph


class AnalysisResult(BaseModel):
    """Result of conversation analysis."""
    topics_found: List[str]
    relationships: List[Dict[str, Any]]
    suggestions: List[str]
    depth_analysis: Dict[str, str]
    graph_stats: Dict[str, Any]


class EulerMCPServer:
    """Main server class handling MCP requests and learning graph operations."""
    
    def __init__(self, db_path: str = "data/graph.db"):
        """
        Initialize the EulerMCP server.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.processor = ConversationProcessor()
        self.graph = LearningGraph(str(self.db_path))
        
        self.logger.info("EulerMCP server initialized successfully")
    
    async def analyze_conversation(self, conversation_text: str, 
                                 conversation_id: str = None) -> AnalysisResult:
        """
        Analyze conversation and update learning graph.
        
        Args:
            conversation_text: Full conversation text to analyze
            conversation_id: Optional conversation identifier
            
        Returns:
            AnalysisResult with extracted topics and suggestions
        """
        try:
            self.logger.info(f"Analyzing conversation (ID: {conversation_id})")
            
            # Step 1: Extract topics using NLP
            topics = await self.processor.extract_topics(conversation_text)
            self.logger.info(f"Extracted {len(topics)} topics")
            
            # Step 2: Add topics to graph and get topic IDs
            topic_ids = []
            for topic in topics:
                topic_id = self.graph.add_topic(
                    name=topic['name'],
                    category=topic['category'], 
                    confidence=topic['confidence'],
                    status='mentioned' if topic['depth'] == 'shallow' else 'learned'
                )
                topic_ids.append(topic_id)
            
            # Step 3: Identify and add relationships
            relationships = await self.processor.identify_relationships(topics)
            
            for rel in relationships:
                # Get topic IDs for relationship
                source_topic = self.graph.get_topic_by_name(rel['source'])
                target_topic = self.graph.get_topic_by_name(rel['target'])
                
                if source_topic and target_topic:
                    self.graph.add_relationship(
                        source_topic['id'],
                        target_topic['id'],
                        relationship_type=rel['type'],
                        weight=rel['weight'],
                        confidence=rel['confidence']
                    )
            
            # Step 4: Generate learning suggestions
            current_topics = [topic['name'] for topic in topics]
            suggestions = await self.graph.suggest_next_topics(
                current_topics=current_topics,
                max_suggestions=5
            )
            
            # Step 5: Get graph statistics
            graph_data = self.graph.get_graph_data()
            
            # Step 6: Store learning session
            if conversation_id:
                await self._store_learning_session(
                    conversation_id, topic_ids, conversation_text
                )
            
            return AnalysisResult(
                topics_found=[t['name'] for t in topics],
                relationships=relationships,
                suggestions=suggestions,
                depth_analysis={t['name']: t['depth'] for t in topics},
                graph_stats=graph_data['stats']
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing conversation: {e}")
            raise
    
    async def get_learning_graph(self, format_type: str = "json") -> Dict[str, Any]:
        """
        Get current learning graph data.
        
        Args:
            format_type: Output format ('json', 'cytoscape', 'networkx')
            
        Returns:
            Graph data in requested format
        """
        try:
            graph_data = self.graph.get_graph_data()
            
            if format_type == "cytoscape":
                # Convert to Cytoscape.js format for web visualization
                return self._convert_to_cytoscape(graph_data)
            elif format_type == "networkx":
                # Return NetworkX-compatible format
                return self._convert_to_networkx_dict(graph_data)
            else:
                # Default JSON format
                return graph_data
                
        except Exception as e:
            self.logger.error(f"Error getting learning graph: {e}")
            return {"error": str(e)}
    
    async def suggest_next_topics(self, current_topics: List[str] = None,
                                max_suggestions: int = 5,
                                categories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get personalized learning suggestions.
        
        Args:
            current_topics: Topics currently known/learning
            max_suggestions: Maximum number of suggestions
            categories: Filter suggestions by categories
            
        Returns:
            List of suggested topics with metadata
        """
        try:
            # Get basic suggestions from graph
            suggestions = await self.graph.suggest_next_topics(
                current_topics or [], max_suggestions
            )
            
            # Enrich suggestions with additional metadata
            enriched_suggestions = []
            for topic_name in suggestions:
                topic_data = self.graph.get_topic_by_name(topic_name)
                if topic_data:
                    # Analyze learning path for this topic
                    path_analysis = self.graph.analyze_learning_path(topic_name)
                    
                    # Filter by categories if specified
                    if categories and topic_data['category'] not in categories:
                        continue
                    
                    enriched_suggestions.append({
                        'name': topic_name,
                        'category': topic_data['category'],
                        'confidence': topic_data['confidence'],
                        'difficulty': topic_data.get('depth_level', 'shallow'),
                        'prerequisites': path_analysis.get('prerequisites', []),
                        'learning_priority': topic_data.get('learning_priority', 0.5),
                        'times_mentioned': topic_data.get('times_mentioned', 1),
                        'estimated_effort': self._estimate_learning_effort(topic_data),
                        'why_suggested': self._generate_suggestion_reason(
                            topic_name, current_topics or []
                        )
                    })
            
            # Sort by learning priority and relevance
            enriched_suggestions.sort(
                key=lambda x: (x['learning_priority'], x['confidence']), 
                reverse=True
            )
            
            return enriched_suggestions[:max_suggestions]
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return []
    
    async def analyze_learning_path(self, target_topic: str) -> Dict[str, Any]:
        """
        Analyze learning path to reach a specific topic.
        
        Args:
            target_topic: Target topic to analyze
            
        Returns:
            Learning path analysis with recommendations
        """
        try:
            return self.graph.analyze_learning_path(target_topic)
        except Exception as e:
            self.logger.error(f"Error analyzing learning path: {e}")
            return {"error": str(e)}
    
    async def update_topic_status(self, topic_name: str, status: str) -> Dict[str, Any]:
        """
        Update the learning status of a topic.
        
        Args:
            topic_name: Name of the topic
            status: New status ('mentioned', 'learned', 'suggested')
            
        Returns:
            Update result with new suggestions
        """
        try:
            success = self.graph.update_topic_status(topic_name, status)
            
            if success:
                # Generate new suggestions based on updated status
                if status == 'learned':
                    # Get topics that depend on this one
                    suggestions = await self.suggest_next_topics(
                        current_topics=[topic_name],
                        max_suggestions=3
                    )
                    
                    return {
                        "success": True,
                        "message": f"Marked '{topic_name}' as {status}",
                        "new_suggestions": suggestions
                    }
                else:
                    return {
                        "success": True, 
                        "message": f"Updated '{topic_name}' status to {status}"
                    }
            else:
                return {
                    "success": False,
                    "message": f"Failed to update topic '{topic_name}'"
                }
                
        except Exception as e:
            self.logger.error(f"Error updating topic status: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_learning_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about learning progress and patterns.
        
        Returns:
            Learning analytics data
        """
        try:
            graph_data = self.graph.get_graph_data()
            
            # Calculate learning metrics
            nodes = graph_data['nodes']
            edges = graph_data['edges']
            
            # Category distribution
            category_counts = {}
            for node in nodes:
                category = node['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Learning progress
            status_counts = {}
            for node in nodes:
                status = node['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Most connected topics (knowledge hubs)
            topic_connections = {}
            for edge in edges:
                source = edge['source']
                target = edge['target']
                topic_connections[source] = topic_connections.get(source, 0) + 1
                topic_connections[target] = topic_connections.get(target, 0) + 1
            
            # Find top knowledge hubs
            top_hubs = sorted(topic_connections.items(), key=lambda x: x[1], reverse=True)[:5]
            hub_names = []
            for node_id, connections in top_hubs:
                for node in nodes:
                    if node['id'] == node_id:
                        hub_names.append({
                            'name': node['name'],
                            'connections': connections,
                            'category': node['category']
                        })
                        break
            
            # Learning velocity (topics learned over time)
            total_learned = status_counts.get('learned', 0)
            total_topics = len(nodes)
            learning_completion = (total_learned / total_topics * 100) if total_topics > 0 else 0
            
            return {
                "overview": {
                    "total_topics": total_topics,
                    "topics_learned": total_learned,
                    "learning_completion_percentage": round(learning_completion, 1),
                    "total_connections": len(edges),
                    "knowledge_areas": len(category_counts)
                },
                "category_distribution": category_counts,
                "learning_progress": status_counts,
                "knowledge_hubs": hub_names,
                "recommendations": self._generate_learning_recommendations(
                    status_counts, category_counts, total_topics
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error generating analytics: {e}")
            return {"error": str(e)}
    
    async def _store_learning_session(self, conversation_id: str, 
                                    topic_ids: List[int], 
                                    conversation_text: str):
        """Store learning session data for analytics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Determine main focus topic (most confident one)
                main_focus = "general"
                if topic_ids:
                    # Get the topic with highest confidence
                    topic_data = conn.execute("""
                        SELECT name FROM topics 
                        WHERE id IN ({}) 
                        ORDER BY confidence DESC 
                        LIMIT 1
                    """.format(','.join(['?'] * len(topic_ids))), topic_ids).fetchone()
                    
                    if topic_data:
                        main_focus = topic_data[0]
                
                # Generate session summary
                session_summary = f"Discussed {len(topic_ids)} topics, main focus: {main_focus}"
                
                # Store session
                conn.execute("""
                    INSERT INTO learning_sessions 
                    (conversation_id, topics_discussed, main_focus, session_summary)
                    VALUES (?, ?, ?, ?)
                """, (conversation_id, json.dumps(topic_ids), main_focus, session_summary))
                
        except Exception as e:
            self.logger.error(f"Error storing learning session: {e}")
    
    def _convert_to_cytoscape(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert graph data to Cytoscape.js format."""
        cytoscape_nodes = []
        cytoscape_edges = []
        
        # Convert nodes
        for node in graph_data['nodes']:
            cytoscape_nodes.append({
                'data': {
                    'id': str(node['id']),
                    'label': node['name'],
                    'category': node['category'],
                    'status': node['status'],
                    'confidence': node['confidence']
                }
            })
        
        # Convert edges
        for edge in graph_data['edges']:
            cytoscape_edges.append({
                'data': {
                    'id': f"{edge['source']}-{edge['target']}",
                    'source': str(edge['source']),
                    'target': str(edge['target']),
                    'type': edge['relationship_type'],
                    'weight': edge['weight']
                }
            })
        
        return {
            'elements': {
                'nodes': cytoscape_nodes,
                'edges': cytoscape_edges
            },
            'stats': graph_data['stats']
        }
    
    def _convert_to_networkx_dict(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to NetworkX-compatible dictionary format."""
        nx_data = {
            'directed': True,
            'multigraph': False,
            'graph': {'name': 'EulerMCP Knowledge Graph'},
            'nodes': [],
            'links': []
        }
        
        # Add nodes
        for node in graph_data['nodes']:
            nx_data['nodes'].append({
                'id': node['id'],
                **node
            })
        
        # Add edges (called 'links' in NetworkX JSON)
        for edge in graph_data['edges']:
            nx_data['links'].append({
                'source': edge['source'],
                'target': edge['target'],
                **edge
            })
        
        return nx_data
    
    def _estimate_learning_effort(self, topic_data: Dict[str, Any]) -> str:
        """Estimate learning effort for a topic."""
        depth = topic_data.get('depth_level', 'shallow')
        
        if depth == 'deep':
            return 'High (2-4 weeks)'
        elif depth == 'medium':
            return 'Medium (1-2 weeks)'
        else:
            return 'Low (few days)'
    
    def _generate_suggestion_reason(self, topic_name: str, current_topics: List[str]) -> str:
        """Generate explanation for why a topic is suggested."""
        if not current_topics:
            return "Good starting point for beginners"
        
        return f"Related to your current learning in: {', '.join(current_topics[:2])}"
    
    def _generate_learning_recommendations(self, status_counts: Dict, 
                                         category_counts: Dict, 
                                         total_topics: int) -> List[str]:
        """Generate learning recommendations based on analytics."""
        recommendations = []
        
        learned_count = status_counts.get('learned', 0)
        mentioned_count = status_counts.get('mentioned', 0)
        
        # Progress-based recommendations
        if learned_count == 0:
            recommendations.append("Start by marking some topics as 'learned' to get personalized suggestions")
        elif learned_count / total_topics < 0.2:
            recommendations.append("Focus on building solid fundamentals in your main area of interest")
        elif mentioned_count > learned_count * 2:
            recommendations.append("Consider deepening your knowledge in topics you've already explored")
        
        # Category-based recommendations
        if len(category_counts) == 1:
            recommendations.append("Try exploring related fields to broaden your knowledge base")
        
        return recommendations


# Create MCP server instance
server = Server("euler-mcp")
euler_server = EulerMCPServer()

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="analyze_conversation",
            description="Analyze conversation text to extract learning topics, relationships, and generate suggestions",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_text": {
                        "type": "string",
                        "description": "The conversation text to analyze"
                    },
                    "conversation_id": {
                        "type": "string", 
                        "description": "Optional conversation identifier"
                    }
                },
                "required": ["conversation_text"]
            }
        ),
        Tool(
            name="get_learning_graph",
            description="Get the current learning knowledge graph in various formats",
            inputSchema={
                "type": "object",
                "properties": {
                    "format_type": {
                        "type": "string",
                        "enum": ["json", "cytoscape", "networkx"],
                        "description": "Output format for the graph data",
                        "default": "json"
                    }
                }
            }
        ),
        Tool(
            name="suggest_next_topics",
            description="Get personalized learning suggestions based on current knowledge",
            inputSchema={
                "type": "object",
                "properties": {
                    "current_topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics currently known or being learned"
                    },
                    "max_suggestions": {
                        "type": "integer",
                        "description": "Maximum number of suggestions to return",
                        "default": 5
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter suggestions by specific categories"
                    }
                }
            }
        ),
        Tool(
            name="analyze_learning_path",
            description="Analyze the learning path required to master a specific topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_topic": {
                        "type": "string",
                        "description": "The topic to analyze learning path for"
                    }
                },
                "required": ["target_topic"]
            }
        ),
        Tool(
            name="update_topic_status",
            description="Update the learning status of a topic (mentioned, learned, suggested)",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic_name": {
                        "type": "string",
                        "description": "Name of the topic to update"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["mentioned", "learned", "suggested"],
                        "description": "New learning status for the topic"
                    }
                },
                "required": ["topic_name", "status"]
            }
        ),
        Tool(
            name="get_learning_analytics",
            description="Get detailed analytics about learning progress and patterns",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Handle tool calls."""
    try:
        if name == "analyze_conversation":
            result = await euler_server.analyze_conversation(
                arguments["conversation_text"],
                arguments.get("conversation_id")
            )
            return [{"type": "text", "text": json.dumps(result.dict(), indent=2)}]
        
        elif name == "get_learning_graph":
            graph_data = await euler_server.get_learning_graph(
                arguments.get("format_type", "json")
            )
            return [{"type": "text", "text": json.dumps(graph_data, indent=2)}]
        
        elif name == "suggest_next_topics":
            suggestions = await euler_server.suggest_next_topics(
                arguments.get("current_topics", []),
                arguments.get("max_suggestions", 5),
                arguments.get("categories")
            )
            return [{"type": "text", "text": json.dumps({"suggestions": suggestions}, indent=2)}]
        
        elif name == "analyze_learning_path":
            path_analysis = await euler_server.analyze_learning_path(
                arguments["target_topic"]
            )
            return [{"type": "text", "text": json.dumps(path_analysis, indent=2)}]
        
        elif name == "update_topic_status":
            result = await euler_server.update_topic_status(
                arguments["topic_name"],
                arguments["status"]
            )
            return [{"type": "text", "text": json.dumps(result, indent=2)}]
        
        elif name == "get_learning_analytics":
            analytics = await euler_server.get_learning_analytics()
            return [{"type": "text", "text": json.dumps(analytics, indent=2)}]
        
        else:
            return [{"type": "text", "text": f"Unknown tool: {name}"}]
    
    except Exception as e:
        return [{"type": "text", "text": f"Error: {str(e)}"}]


def main():
    """Main entry point."""
    asyncio.run(stdio_server(server))


if __name__ == "__main__":
    main()