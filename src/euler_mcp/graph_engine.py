"""
GraphEngine: Manages the learning knowledge graph using NetworkX and SQLite.

This module handles:
1. Building and maintaining the knowledge graph
2. Suggesting next learning topics
3. Analyzing learning paths and gaps
4. Exporting graph data for visualization
"""

import sqlite3
import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
import json
import logging
from pathlib import Path


class LearningGraph:
    """Main class for managing the learning knowledge graph."""
    
    def __init__(self, db_path: str = "data/graph.db"):
        """
        Initialize the learning graph with database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize NetworkX graph for in-memory operations
        self.graph = nx.DiGraph()  # Directed graph for learning dependencies
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize database and load existing graph
        self._init_database()
        self._load_graph_from_db()
    
    def _init_database(self):
        """Initialize database with required tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    category TEXT,
                    description TEXT,
                    confidence REAL DEFAULT 0.0,
                    depth_level TEXT CHECK(depth_level IN ('shallow', 'medium', 'deep')) DEFAULT 'shallow',
                    status TEXT CHECK(status IN ('learned', 'mentioned', 'suggested')) DEFAULT 'mentioned',
                    first_mentioned TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_discussed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    times_mentioned INTEGER DEFAULT 1,
                    learning_priority REAL DEFAULT 0.5
                );
                
                CREATE TABLE IF NOT EXISTS topic_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_topic_id INTEGER,
                    target_topic_id INTEGER,
                    relationship_type TEXT CHECK(relationship_type IN ('prerequisite', 'related', 'extension', 'alternative')),
                    weight REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_topic_id) REFERENCES topics(id),
                    FOREIGN KEY (target_topic_id) REFERENCES topics(id),
                    UNIQUE(source_topic_id, target_topic_id, relationship_type)
                );
                
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    topics_discussed TEXT, -- JSON array of topic IDs
                    main_focus TEXT,
                    session_summary TEXT,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    duration_minutes INTEGER DEFAULT 0
                );
                
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create indexes for better performance
                CREATE INDEX IF NOT EXISTS idx_topics_category ON topics(category);
                CREATE INDEX IF NOT EXISTS idx_topics_status ON topics(status);
                CREATE INDEX IF NOT EXISTS idx_topics_priority ON topics(learning_priority DESC);
                CREATE INDEX IF NOT EXISTS idx_relationships_source ON topic_relationships(source_topic_id);
                CREATE INDEX IF NOT EXISTS idx_relationships_target ON topic_relationships(target_topic_id);
            """)
    
    def _load_graph_from_db(self):
        """Load existing graph data from database into NetworkX."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Load topics as nodes
                topics = conn.execute("SELECT * FROM topics").fetchall()
                for topic in topics:
                    self.graph.add_node(
                        topic['id'],
                        name=topic['name'],
                        category=topic['category'],
                        confidence=topic['confidence'],
                        depth_level=topic['depth_level'],
                        status=topic['status'],
                        last_discussed=topic['last_discussed'],
                        times_mentioned=topic['times_mentioned'],
                        learning_priority=topic['learning_priority']
                    )
                
                # Load relationships as edges
                relationships = conn.execute("""
                    SELECT r.*, 
                           s.name as source_name, 
                           t.name as target_name
                    FROM topic_relationships r
                    JOIN topics s ON r.source_topic_id = s.id
                    JOIN topics t ON r.target_topic_id = t.id
                """).fetchall()
                
                for rel in relationships:
                    self.graph.add_edge(
                        rel['source_topic_id'],
                        rel['target_topic_id'],
                        relationship_type=rel['relationship_type'],
                        weight=rel['weight'],
                        confidence=rel['confidence']
                    )
                    
                self.logger.info(f"Loaded graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
                
        except Exception as e:
            self.logger.error(f"Error loading graph from database: {e}")
            # Initialize empty graph if loading fails
            self.graph = nx.DiGraph()
    
    def add_topic(self, name: str, category: str = 'general', confidence: float = 0.5, 
                  status: str = 'mentioned', description: str = None) -> int:
        """
        Add a new topic to the graph or update existing one.
        
        Args:
            name: Topic name
            category: Topic category
            confidence: Confidence score (0-1)
            status: Topic status ('mentioned', 'learned', 'suggested')
            description: Optional topic description
            
        Returns:
            Topic ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if topic already exists
                existing = conn.execute(
                    "SELECT id, times_mentioned FROM topics WHERE name = ?", 
                    (name,)
                ).fetchone()
                
                if existing:
                    # Update existing topic
                    topic_id = existing[0]
                    new_times_mentioned = existing[1] + 1
                    
                    conn.execute("""
                        UPDATE topics 
                        SET confidence = ?, 
                            last_discussed = CURRENT_TIMESTAMP,
                            times_mentioned = ?,
                            status = CASE WHEN ? = 'learned' THEN 'learned' ELSE status END
                        WHERE id = ?
                    """, (confidence, new_times_mentioned, status, topic_id))
                    
                    # Update NetworkX graph
                    if topic_id in self.graph:
                        self.graph.nodes[topic_id].update({
                            'confidence': confidence,
                            'last_discussed': datetime.now().isoformat(),
                            'times_mentioned': new_times_mentioned,
                            'status': status if status == 'learned' else self.graph.nodes[topic_id]['status']
                        })
                else:
                    # Insert new topic
                    cursor = conn.execute("""
                        INSERT INTO topics (name, category, confidence, status, description)
                        VALUES (?, ?, ?, ?, ?)
                    """, (name, category, confidence, status, description))
                    
                    topic_id = cursor.lastrowid
                    
                    # Add to NetworkX graph
                    self.graph.add_node(
                        topic_id,
                        name=name,
                        category=category,
                        confidence=confidence,
                        status=status,
                        description=description,
                        last_discussed=datetime.now().isoformat(),
                        times_mentioned=1,
                        learning_priority=confidence
                    )
                
                self.logger.info(f"Added/updated topic: {name} (ID: {topic_id})")
                return topic_id
                
        except Exception as e:
            self.logger.error(f"Error adding topic {name}: {e}")
            raise
    
    def add_relationship(self, source_topic_id: int, target_topic_id: int, 
                        relationship_type: str = 'related', weight: float = 0.5, 
                        confidence: float = 0.5) -> bool:
        """
        Add a relationship between two topics.
        
        Args:
            source_topic_id: Source topic ID
            target_topic_id: Target topic ID  
            relationship_type: Type of relationship
            weight: Relationship strength (0-1)
            confidence: Confidence in relationship (0-1)
            
        Returns:
            True if relationship was added successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Insert relationship (ON CONFLICT IGNORE prevents duplicates)
                conn.execute("""
                    INSERT OR IGNORE INTO topic_relationships 
                    (source_topic_id, target_topic_id, relationship_type, weight, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (source_topic_id, target_topic_id, relationship_type, weight, confidence))
                
                # Add edge to NetworkX graph
                self.graph.add_edge(
                    source_topic_id, 
                    target_topic_id,
                    relationship_type=relationship_type,
                    weight=weight,
                    confidence=confidence
                )
                
                self.logger.debug(f"Added relationship: {source_topic_id} -> {target_topic_id} ({relationship_type})")
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding relationship: {e}")
            return False
    
    def get_topic_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get topic information by name.
        
        Args:
            name: Topic name
            
        Returns:
            Topic information dict or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                topic = conn.execute(
                    "SELECT * FROM topics WHERE name = ?", 
                    (name,)
                ).fetchone()
                
                if topic:
                    return dict(topic)
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting topic {name}: {e}")
            return None
    
    async def suggest_next_topics(self, current_topics: List[str] = None, 
                                 max_suggestions: int = 5) -> List[str]:
        """
        Suggest next topics to learn based on current knowledge and graph analysis.
        
        Args:
            current_topics: List of currently known topics
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggested topic names
        """
        try:
            suggestions = []
            
            if not current_topics:
                # If no current topics, suggest popular starting points
                suggestions = self._get_popular_starting_topics(max_suggestions)
            else:
                # Get topic IDs for current topics
                current_topic_ids = []
                for topic_name in current_topics:
                    topic = self.get_topic_by_name(topic_name)
                    if topic:
                        current_topic_ids.append(topic['id'])
                
                if current_topic_ids:
                    # Find connected topics that haven't been learned
                    candidate_topics = self._find_next_learning_candidates(current_topic_ids)
                    
                    # Score and rank suggestions
                    scored_suggestions = self._score_learning_suggestions(
                        candidate_topics, current_topic_ids
                    )
                    
                    # Return top suggestions
                    suggestions = [topic['name'] for topic in scored_suggestions[:max_suggestions]]
            
            self.logger.info(f"Generated {len(suggestions)} suggestions")
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return []
    
    def _get_popular_starting_topics(self, limit: int) -> List[str]:
        """Get popular topics that are good starting points for learning."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                topics = conn.execute("""
                    SELECT name FROM topics 
                    WHERE status IN ('mentioned', 'learned')
                    AND category IN ('programming_languages', 'web_development', 'data_science_ml_ai')
                    ORDER BY times_mentioned DESC, confidence DESC
                    LIMIT ?
                """, (limit,)).fetchall()
                
                return [topic[0] for topic in topics]
                
        except Exception as e:
            self.logger.error(f"Error getting starting topics: {e}")
            return ['python', 'javascript', 'html', 'css']  # Fallback defaults
    
    def _find_next_learning_candidates(self, current_topic_ids: List[int]) -> List[int]:
        """Find topics that are connected to current topics but not yet learned."""
        candidates = set()
        
        for topic_id in current_topic_ids:
            # Get topics that this topic points to (what you could learn next)
            successors = list(self.graph.successors(topic_id))
            candidates.update(successors)
            
            # Get topics that point to this topic (prerequisites you might have missed)
            predecessors = list(self.graph.predecessors(topic_id))
            candidates.update(predecessors)
        
        # Filter out topics that are already learned or currently known
        filtered_candidates = []
        for candidate_id in candidates:
            if candidate_id not in current_topic_ids:
                node_data = self.graph.nodes.get(candidate_id, {})
                if node_data.get('status', 'mentioned') != 'learned':
                    filtered_candidates.append(candidate_id)
        
        return filtered_candidates
    
    def _score_learning_suggestions(self, candidate_ids: List[int], 
                                   current_topic_ids: List[int]) -> List[Dict[str, Any]]:
        """Score and rank learning suggestions based on various factors."""
        scored_topics = []
        
        for candidate_id in candidate_ids:
            node_data = self.graph.nodes.get(candidate_id, {})
            score = 0.0
            
            # Base score from topic properties
            confidence = node_data.get('confidence', 0.5)
            times_mentioned = node_data.get('times_mentioned', 1)
            
            # Confidence contributes to score
            score += confidence * 0.3
            
            # Popularity (times mentioned) contributes
            score += min(times_mentioned / 10.0, 0.3)  # Cap at 0.3
            
            # Connection strength to current topics
            connection_score = 0.0
            total_connections = 0
            
            for current_id in current_topic_ids:
                # Check edge from current to candidate
                if self.graph.has_edge(current_id, candidate_id):
                    edge_data = self.graph.edges[current_id, candidate_id]
                    connection_score += edge_data.get('weight', 0.5)
                    total_connections += 1
                
                # Check edge from candidate to current
                if self.graph.has_edge(candidate_id, current_id):
                    edge_data = self.graph.edges[candidate_id, current_id]
                    connection_score += edge_data.get('weight', 0.5)
                    total_connections += 1
            
            if total_connections > 0:
                score += (connection_score / total_connections) * 0.4
            
            scored_topics.append({
                'id': candidate_id,
                'name': node_data.get('name', 'Unknown'),
                'category': node_data.get('category', 'general'),
                'score': score,
                'confidence': confidence,
                'times_mentioned': times_mentioned
            })
        
        # Sort by score descending
        return sorted(scored_topics, key=lambda x: x['score'], reverse=True)
    
    def get_graph_data(self) -> Dict[str, Any]:
        """
        Export graph data for visualization in Gradio interface.
        
        Returns:
            Dictionary containing nodes and edges data
        """
        try:
            nodes = []
            edges = []
            
            # Export nodes
            for node_id, node_data in self.graph.nodes(data=True):
                nodes.append({
                    'id': node_id,
                    'name': node_data.get('name', 'Unknown'),
                    'category': node_data.get('category', 'general'),
                    'confidence': node_data.get('confidence', 0.5),
                    'status': node_data.get('status', 'mentioned'),
                    'depth_level': node_data.get('depth_level', 'shallow'),
                    'times_mentioned': node_data.get('times_mentioned', 1),
                    'last_discussed': node_data.get('last_discussed', ''),
                    'learning_priority': node_data.get('learning_priority', 0.5)
                })
            
            # Export edges
            for source, target, edge_data in self.graph.edges(data=True):
                edges.append({
                    'source': source,
                    'target': target,
                    'relationship_type': edge_data.get('relationship_type', 'related'),
                    'weight': edge_data.get('weight', 0.5),
                    'confidence': edge_data.get('confidence', 0.5)
                })
            
            return {
                'nodes': nodes,
                'edges': edges,
                'stats': {
                    'total_topics': len(nodes),
                    'total_connections': len(edges),
                    'categories': len(set(node['category'] for node in nodes)),
                    'learned_topics': len([n for n in nodes if n['status'] == 'learned']),
                    'mentioned_topics': len([n for n in nodes if n['status'] == 'mentioned']),
                    'suggested_topics': len([n for n in nodes if n['status'] == 'suggested'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting graph data: {e}")
            return {'nodes': [], 'edges': [], 'stats': {}}
    
    def analyze_learning_path(self, target_topic: str) -> Dict[str, Any]:
        """
        Analyze the learning path to reach a target topic.
        
        Args:
            target_topic: Name of the target topic
            
        Returns:
            Analysis of learning path including prerequisites and recommendations
        """
        try:
            target = self.get_topic_by_name(target_topic)
            if not target:
                return {'error': f"Topic '{target_topic}' not found in graph"}
            
            target_id = target['id']
            
            # Find all paths to the target topic
            prerequisites = []
            learned_topics = []
            
            # Get all topics with 'learned' status
            with sqlite3.connect(self.db_path) as conn:
                learned = conn.execute(
                    "SELECT id, name FROM topics WHERE status = 'learned'"
                ).fetchall()
                learned_topics = [{'id': t[0], 'name': t[1]} for t in learned]
            
            # Find shortest paths from learned topics to target
            shortest_paths = []
            for learned_topic in learned_topics:
                try:
                    if nx.has_path(self.graph, learned_topic['id'], target_id):
                        path = nx.shortest_path(self.graph, learned_topic['id'], target_id)
                        if len(path) > 1:  # Exclude direct connection
                            shortest_paths.append({
                                'from': learned_topic['name'],
                                'path_length': len(path) - 1,
                                'path_nodes': [self.graph.nodes[node_id]['name'] for node_id in path]
                            })
                except nx.NetworkXNoPath:
                    continue
            
            # Sort paths by length
            shortest_paths.sort(key=lambda x: x['path_length'])
            
            # Find immediate prerequisites
            predecessors = list(self.graph.predecessors(target_id))
            for pred_id in predecessors:
                pred_data = self.graph.nodes.get(pred_id, {})
                edge_data = self.graph.edges.get((pred_id, target_id), {})
                if edge_data.get('relationship_type') == 'prerequisite':
                    prerequisites.append({
                        'name': pred_data.get('name', 'Unknown'),
                        'status': pred_data.get('status', 'mentioned'),
                        'confidence': edge_data.get('confidence', 0.5)
                    })
            
            return {
                'target_topic': target_topic,
                'target_status': target['status'],
                'prerequisites': prerequisites,
                'shortest_paths': shortest_paths[:3],  # Top 3 shortest paths
                'recommendations': self._generate_path_recommendations(target_id, prerequisites)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing learning path for {target_topic}: {e}")
            return {'error': f"Error analyzing learning path: {str(e)}"}
    
    def _generate_path_recommendations(self, target_id: int, prerequisites: List[Dict]) -> List[str]:
        """Generate recommendations for learning path to target topic."""
        recommendations = []
        
        # Check if there are unlearned prerequisites
        unlearned_prereqs = [p for p in prerequisites if p['status'] != 'learned']
        if unlearned_prereqs:
            recommendations.append(
                f"Focus on these prerequisites first: {', '.join([p['name'] for p in unlearned_prereqs])}"
            )
        
        # Check topic difficulty
        target_data = self.graph.nodes.get(target_id, {})
        depth = target_data.get('depth_level', 'shallow')
        if depth == 'deep':
            recommendations.append("This is an advanced topic. Make sure you have solid fundamentals.")
        
        # Check for alternative learning paths
        predecessors = list(self.graph.predecessors(target_id))
        if len(predecessors) > 1:
            recommendations.append("Multiple learning paths available. Choose based on your background.")
        
        return recommendations
    
    def update_topic_status(self, topic_name: str, new_status: str) -> bool:
        """
        Update the learning status of a topic.
        
        Args:
            topic_name: Name of the topic
            new_status: New status ('mentioned', 'learned', 'suggested')
            
        Returns:
            True if update was successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("""
                    UPDATE topics 
                    SET status = ?, last_discussed = CURRENT_TIMESTAMP
                    WHERE name = ?
                """, (new_status, topic_name))
                
                if result.rowcount > 0:
                    # Update NetworkX graph
                    topic = self.get_topic_by_name(topic_name)
                    if topic and topic['id'] in self.graph:
                        self.graph.nodes[topic['id']]['status'] = new_status
                    
                    self.logger.info(f"Updated {topic_name} status to {new_status}")
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating topic status: {e}")
            return False