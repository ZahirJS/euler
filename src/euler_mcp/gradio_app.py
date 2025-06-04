"""
Gradio interface for EulerMCP Learning Knowledge Graph.

This module provides a web-based interface for:
1. Visualizing the learning knowledge graph
2. Analyzing conversations and extracting topics
3. Getting personalized learning suggestions
4. Tracking learning progress and analytics
"""

import gradio as gr
import json
import asyncio
import logging
from typing import Dict, List, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import networkx as nx
import pandas as pd

from .conversation_processor import ConversationProcessor
from .graph_engine import LearningGraph
from .config import TOPIC_CATEGORIES  # ← Importar las categorías reales


class EulerMCPInterface:
    """Main interface class for the Gradio web application."""
    
    def __init__(self, db_path: str = "data/graph.db"):
        """
        Initialize the Gradio interface.
        
        Args:
            db_path: Path to the SQLite database
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.processor = ConversationProcessor()
        self.graph = LearningGraph(db_path)
        
        # Generate category colors dynamically from config
        self.category_colors = self._generate_category_colors()
        self.category_emojis = self._generate_category_emojis()
        
        self.logger.info("EulerMCP Gradio interface initialized")
    
    def _generate_category_colors(self) -> Dict[str, str]:
        """Generate colors for all categories automatically."""
        # Predefined color palette
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD',
            '#9B59B6', '#E74C3C', '#F39C12', '#3498DB', '#E67E22', '#1ABC9C',
            '#2ECC71', '#8E44AD', '#34495E', '#16A085', '#D35400', '#F1C40F',
            '#27AE60', '#E91E63', '#95A5A6', '#C0392B', '#8E44AD', '#F39C12'
        ]
        
        category_colors = {}
        categories = list(TOPIC_CATEGORIES.keys())
        
        for i, category in enumerate(categories):
            color_index = i % len(colors)
            category_colors[category] = colors[color_index]
        
        # Ensure 'general' always has a default color
        category_colors['general'] = '#95A5A6'
        
        return category_colors
    
    def _generate_category_emojis(self) -> Dict[str, str]:
        """Generate emojis for all categories automatically."""
        emoji_mapping = {
            'programming_languages': '💻',
            'web_development': '🌐',
            'mobile_development': '📱',
            'data_science_ml_ai': '📊',
            'llm_ai_models_2025': '🤖',
            'package_managers': '📦',
            'databases': '🗄️',
            'cloud_computing': '☁️',
            'devops_infrastructure': '🔧',
            'cybersecurity': '🔒',
            'networking': '🌍',
            'operating_systems': '💾',
            'algorithms_data_structures': '🧮',
            'software_engineering': '⚙️',
            'game_development': '🎮',
            'computer_graphics': '🎨',
            'blockchain_crypto': '₿',
            'iot_embedded': '📡',
            'browsers_web_tools': '🌍',
            'general': '🔧'
        }
        
        # For any category not explicitly mapped, use a default
        for category in TOPIC_CATEGORIES.keys():
            if category not in emoji_mapping:
                emoji_mapping[category] = '🔧'
        
        return emoji_mapping
    
    async def analyze_conversation_interface(self, conversation_text: str, 
                                           conversation_id: str = None) -> Tuple[str, str, str, str]:
        """
        Interface wrapper for conversation analysis.
        
        Args:
            conversation_text: Text to analyze
            conversation_id: Optional conversation ID
            
        Returns:
            Tuple of (topics_json, relationships_json, suggestions_text, graph_html)
        """
        try:
            if not conversation_text.strip():
                return "❌ Please enter some conversation text", "", "", ""
            
            # Extract topics
            topics = await self.processor.extract_topics(conversation_text)
            
            # Add topics to graph
            topic_ids = []
            for topic in topics:
                topic_id = self.graph.add_topic(
                    name=topic['name'],
                    category=topic['category'],
                    confidence=topic['confidence'],
                    status='mentioned'
                )
                topic_ids.append(topic_id)
            
            # Identify relationships
            relationships = await self.processor.identify_relationships(topics)
            
            # Add relationships to graph
            for rel in relationships:
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
            
            # Generate suggestions
            current_topics = [topic['name'] for topic in topics]
            suggestions = await self.graph.suggest_next_topics(current_topics, 5)
            
            # Format outputs
            topics_output = self._format_topics_output(topics)
            relationships_output = self._format_relationships_output(relationships)
            suggestions_output = self._format_suggestions_output(suggestions)
            
            # Generate graph visualization
            graph_html = self._create_graph_visualization()
            
            return topics_output, relationships_output, suggestions_output, graph_html
            
        except Exception as e:
            self.logger.error(f"Error in conversation analysis: {e}")
            return f"❌ Error: {str(e)}", "", "", ""
    
    def _format_topics_output(self, topics: List[Dict[str, Any]]) -> str:
        """Format topics for display."""
        if not topics:
            return "No topics found."
        
        output = f"## 🎯 Topics Found ({len(topics)})\n\n"
        
        for i, topic in enumerate(topics, 1):
            confidence_bar = "🟩" * int(topic['confidence'] * 10)
            depth_emoji = {"shallow": "📖", "medium": "📚", "deep": "🎓"}.get(topic['depth'], "📖")
            category_emoji = self.category_emojis.get(topic['category'], "🔧")
            
            output += f"**{i}. {topic['name']}** {category_emoji}\n"
            output += f"   - Category: {topic['category']}\n"
            output += f"   - Confidence: {confidence_bar} ({topic['confidence']:.2f})\n"
            output += f"   - Depth: {depth_emoji} {topic['depth']}\n"
            output += f"   - Frequency: {topic['frequency']}\n\n"
        
        return output
    
    def _format_relationships_output(self, relationships: List[Dict[str, Any]]) -> str:
        """Format relationships for display."""
        if not relationships:
            return "No relationships found."
        
        output = f"## 🕸️ Relationships Found ({len(relationships)})\n\n"
        
        # Group by relationship type
        by_type = {}
        for rel in relationships:
            rel_type = rel['type']
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append(rel)
        
        type_emojis = {
            "prerequisite": "🔗", "related": "↔️", "extension": "📈", "alternative": "🔄"
        }
        
        for rel_type, rels in by_type.items():
            emoji = type_emojis.get(rel_type, "➡️")
            output += f"### {emoji} {rel_type.title()} ({len(rels)})\n"
            
            for rel in rels:
                strength = "🟩" * int(rel['weight'] * 5)
                output += f"- **{rel['source']}** → **{rel['target']}** {strength} ({rel['weight']:.3f})\n"
            
            output += "\n"
        
        return output
    
    def _format_suggestions_output(self, suggestions: List[str]) -> str:
        """Format learning suggestions for display."""
        if not suggestions:
            return "No suggestions available. Try analyzing more conversations!"
        
        output = f"## 💡 Next Learning Suggestions ({len(suggestions)})\n\n"
        
        suggestion_emojis = ["🎯", "🚀", "⭐", "💎", "🔥"]
        
        for i, suggestion in enumerate(suggestions):
            emoji = suggestion_emojis[i % len(suggestion_emojis)]
            output += f"{emoji} **{suggestion}**\n"
            
            # Get additional info about the topic
            topic_data = self.graph.get_topic_by_name(suggestion)
            if topic_data:
                output += f"   - Category: {topic_data['category']}\n"
                output += f"   - Times mentioned: {topic_data.get('times_mentioned', 1)}\n"
            output += "\n"
        
        return output
    
    def _create_graph_visualization(self) -> str:
        """Create an interactive graph visualization using Plotly."""
        try:
            graph_data = self.graph.get_graph_data()
            nodes = graph_data['nodes']
            edges = graph_data['edges']
            
            # DEBUG: Agregar esto
            print(f"DEBUG: Found {len(nodes)} nodes and {len(edges)} edges")
            if nodes:
                print(f"DEBUG: Sample node: {nodes[0]}")

            if not nodes:
                return "<div style='text-align: center; padding: 50px;'>No topics in graph yet. Analyze some conversations first!</div>"
            
            # Create NetworkX graph for layout
            G = nx.DiGraph()
            
            # Add nodes
            for node in nodes:
                G.add_node(node['id'], **node)
            
            # Add edges
            for edge in edges:
                G.add_edge(edge['source'], edge['target'], **edge)
            
            print(f"DEBUG: NetworkX graph created with {len(G.nodes)} nodes, {len(G.edges)} edges")

            # Calculate layout
            try:
                pos = nx.spring_layout(G, k=3, iterations=50)
            except:
                # Fallback to random layout if spring fails
                pos = nx.random_layout(G)
            
            # Prepare data for Plotly
            edge_x = []
            edge_y = []
            edge_info = []
            
            for edge in edges:
                x0, y0 = pos[edge['source']]
                x1, y1 = pos[edge['target']]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_info.append(f"{edge['relationship_type']}: {edge['weight']:.2f}")
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='rgba(125,125,125,0.5)'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Prepare node data
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []
            node_info = []
            
            # Color mapping for categories (matching your topic_categories.py)
            # Now loaded dynamically from config!
            
            # Status colors
            status_colors = {
                'learned': '#00D084',
                'mentioned': '#FFB347', 
                'suggested': '#87CEEB'
            }
            
            for node in nodes:
                x, y = pos[node['id']]
                node_x.append(x)
                node_y.append(y)

                safe_name = self._safe_string(node['name'])
                node_text.append(safe_name)
                
                # Use status color with category as fallback
                color = status_colors.get(node['status'], 
                        self.category_colors.get(node['category'], '#95A5A6'))
                node_color.append(color)
                
                # Size based on confidence and times mentioned
                size = max(10, min(30, 10 + node['confidence'] * 15 + node.get('times_mentioned', 1) * 2))
                node_size.append(size)
                
                # Hover info
                info = f"<b>{self._safe_string(safe_name)}</b><br>"
                info += f"Category: {self._safe_string(node['category'])}<br>"
                info += f"Status: {self._safe_string(node['status'])}<br>"
                info += f"Confidence: {node['confidence']:.2f}<br>"
                info += f"Times mentioned: {node.get('times_mentioned', 1)}<br>"
                info += f"Last discussed: {self._safe_string(str(node.get('last_discussed', 'Never'))[:10])}"
                node_info.append(info)
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                hovertext=node_info,
                text=node_text,
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=2, color="white"),
                    sizemode='diameter'
                )
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text='EulerMCP Learning Knowledge Graph',  # ← SIN EMOJI
                            x=0.5,
                            font=dict(size=20)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Node size = confidence × mentions | Colors = learning status",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(color="gray", size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    ))
            
            # Convert to HTML
            html_result = fig.to_html(include_plotlyjs='cdn', div_id="knowledge-graph")
            print(f"DEBUG: Generated HTML length: {len(html_result)}")

            # AGREGAR ESTE DEBUG ESPECÍFICO:
            print("DEBUG: HTML contains plotly:", "plotly" in html_result.lower())
            print("DEBUG: HTML contains data:", "data" in html_result.lower())
            print("DEBUG: HTML contains nodes:", len([line for line in html_result.split('\n') if 'node' in line.lower()]))
            print("DEBUG: HTML first 500 chars:", html_result[:500])
            print("DEBUG: Node positions sample:", list(pos.items())[:3] if pos else "No positions")
            print("DEBUG: Node data sample:", node_x[:5], node_y[:5], node_text[:3])
            print(f"DEBUG: Node positions range - X: {min(node_x) if node_x else 'none'} to {max(node_x) if node_x else 'none'}")
            print(f"DEBUG: Node positions range - Y: {min(node_y) if node_y else 'none'} to {max(node_y) if node_y else 'none'}")
            print(f"DEBUG: Node sizes: {node_size[:5]}")
            print(f"DEBUG: Node colors: {node_color[:3]}")
            return html_result
            
            
        except Exception as e:
            self.logger.error(f"Error creating graph visualization: {e}")
            return f"<div style='color: red;'>Error creating visualization: {str(e)}</div>"
    
    def _safe_string(self, value) -> str:
        """Safely convert any value to string for display."""
        if value is None:
            return ""
        elif isinstance(value, bytes):
            try:
                return value.decode('utf-8')
            except UnicodeDecodeError:
                return value.decode('utf-8', errors='replace')
        elif isinstance(value, str):
            # Remove any problematic characters that could cause formatting issues
            return value.encode('ascii', errors='ignore').decode('ascii') if len(value) > 100 else value
        else:
            return str(value)

    def get_analytics_data(self) -> Tuple[str, str]:
        """Get learning analytics and create charts."""
        try:
            graph_data = self.graph.get_graph_data()
            stats = graph_data['stats']
            nodes = graph_data['nodes']
            
            # Create overview text
            overview = f"""
            ## 📊 Learning Analytics Dashboard
            
            ### Overview
            - **Total Topics**: {stats['total_topics']}
            - **Topics Learned**: {stats['learned_topics']} 
            - **Topics Mentioned**: {stats['mentioned_topics']}
            - **Suggested Topics**: {stats['suggested_topics']}
            - **Total Connections**: {stats['total_connections']}
            - **Knowledge Areas**: {stats['categories']}
            
            ### Progress
            """
            
            if stats['total_topics'] > 0:
                completion = (stats['learned_topics'] / stats['total_topics']) * 100
                overview += f"- **Learning Completion**: {completion:.1f}%\n"
                
                progress_bar = "🟩" * int(completion / 10) + "⬜" * (10 - int(completion / 10))
                overview += f"- **Progress Bar**: {progress_bar}\n"
            
            # Category distribution chart
            category_data = {}
            for node in nodes:
                cat = node['category']
                category_data[cat] = category_data.get(cat, 0) + 1
            
            if category_data:
                # Create pie chart for categories
                fig_cat = px.pie(
                    values=list(category_data.values()),
                    names=list(category_data.keys()),
                    title="📚 Topics by Category",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_cat.update_traces(textposition='inside', textinfo='percent+label')
                fig_cat.update_layout(
                    font=dict(size=14),
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01)
                )
                
                charts_html = fig_cat.to_html(include_plotlyjs='cdn', div_id="category-chart")
            else:
                charts_html = "<div>No data available for charts</div>"
            
            return overview, charts_html
            
        except Exception as e:
            self.logger.error(f"Error getting analytics: {e}")
            return f"Error: {str(e)}", ""
    
    def update_topic_status_interface(self, topic_name: str, new_status: str) -> str:
        """Interface for updating topic status."""
        try:
            if not topic_name.strip():
                return "❌ Please enter a topic name"
            
            success = self.graph.update_topic_status(topic_name, new_status)
            
            if success:
                return f"✅ Updated '{topic_name}' status to '{new_status}'"
            else:
                return f"❌ Topic '{topic_name}' not found in graph"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def search_topics_interface(self, search_query: str) -> str:
        """Search for topics in the graph."""
        try:
            if not search_query.strip():
                return "Enter a search term"
            
            graph_data = self.graph.get_graph_data()
            nodes = graph_data['nodes']
            
            # Simple text search
            matching_topics = []
            search_lower = search_query.lower()
            
            for node in nodes:
                if search_lower in node['name'].lower() or search_lower in node['category'].lower():
                    matching_topics.append(node)
            
            if not matching_topics:
                return f"No topics found matching '{search_query}'"
            
            result = f"## 🔍 Search Results for '{search_query}' ({len(matching_topics)} found)\n\n"
            
            for topic in matching_topics:
                status_emoji = {"learned": "✅", "mentioned": "📝", "suggested": "💡"}.get(topic['status'], "❓")
                result += f"{status_emoji} **{topic['name']}** ({topic['category']})\n"
                result += f"   - Status: {topic['status']}\n"
                result += f"   - Confidence: {topic['confidence']:.2f}\n"
                result += f"   - Times mentioned: {topic.get('times_mentioned', 1)}\n\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error: {str(e)}"


def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Initialize the interface
    interface = EulerMCPInterface()
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
    }
    .gr-textbox textarea {
        border-radius: 8px;
    }
    #knowledge-graph {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    """
    
    with gr.Blocks(
        title="🧠 EulerMCP - Learning Knowledge Graph",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as app:
        
        gr.HTML("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🧠 EulerMCP - Learning Knowledge Graph</h1>
            <p style='font-size: 18px; color: #666;'>
                Transform your learning conversations into interactive knowledge maps
            </p>
        </div>
        """)
        
        with gr.Tabs():
            
            # Tab 1: Conversation Analysis
            with gr.Tab("💬 Analyze Conversation"):
                gr.Markdown("## Analyze your learning conversations to extract topics and build knowledge graphs")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        conversation_input = gr.Textbox(
                            label="Conversation Text",
                            placeholder="Paste your learning conversation here...",
                            lines=8,
                            max_lines=20
                        )
                        conversation_id_input = gr.Textbox(
                            label="Conversation ID (optional)",
                            placeholder="e.g., typescript-learning-session-1"
                        )
                        analyze_btn = gr.Button("🔍 Analyze Conversation", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 💡 Tips for better analysis:
                        - Include technical discussions
                        - Mention specific technologies
                        - Ask questions about concepts
                        - Discuss learning goals
                        
                        ### Example conversation:
                        *"I'm learning TypeScript and I'm confused about interfaces vs types. 
                        Can you explain generics and how they work with React components?"*
                        """)
                
                with gr.Row():
                    with gr.Column():
                        topics_output = gr.Markdown(label="📚 Extracted Topics")
                    with gr.Column():
                        relationships_output = gr.Markdown(label="🔗 Topic Relationships")
                
                suggestions_output = gr.Markdown(label="💡 Learning Suggestions")
                
                analyze_btn.click(
                    interface.analyze_conversation_interface,
                    inputs=[conversation_input, conversation_id_input],
                    outputs=[topics_output, relationships_output, suggestions_output, gr.HTML(visible=False)]
                )
            
            # Tab 2: Knowledge Graph Visualization  
            with gr.Tab("🕸️ Knowledge Graph"):
                gr.Markdown("## Interactive visualization of your learning knowledge graph")
                
                with gr.Row():
                    refresh_graph_btn = gr.Button("🔄 Refresh Graph", variant="secondary")
                    gr.Markdown("*Graph updates automatically when you analyze conversations*")
                
                graph_html = gr.HTML(
                    value=interface._create_graph_visualization(),
                    label="Knowledge Graph"
                )
                
                refresh_graph_btn.click(
                lambda: interface._create_graph_visualization(),
                outputs=[graph_html]
)
            
            # Tab 3: Learning Analytics
            with gr.Tab("📊 Analytics"):
                gr.Markdown("## Track your learning progress and discover patterns")
                
                with gr.Row():
                    refresh_analytics_btn = gr.Button("📈 Refresh Analytics", variant="secondary")
                
                with gr.Row():
                    with gr.Column():
                        analytics_text = gr.Markdown()
                    with gr.Column():
                        analytics_charts = gr.HTML()
                
                # Load initial analytics
                initial_analytics = interface.get_analytics_data()
                analytics_text.value = initial_analytics[0]
                analytics_charts.value = initial_analytics[1]
                
                refresh_analytics_btn.click(
                    interface.get_analytics_data,
                    outputs=[analytics_text, analytics_charts]
                )
            
            # Tab 4: Topic Management
            with gr.Tab("🎯 Manage Topics"):
                gr.Markdown("## Update topic status and search your knowledge base")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Update Topic Status")
                        topic_name_input = gr.Textbox(
                            label="Topic Name",
                            placeholder="e.g., typescript, react hooks, machine learning"
                        )
                        status_dropdown = gr.Dropdown(
                            choices=["mentioned", "learned", "suggested"],
                            label="New Status",
                            value="learned"
                        )
                        update_btn = gr.Button("✅ Update Status", variant="primary")
                        status_output = gr.Markdown()
                    
                    with gr.Column():
                        gr.Markdown("### Search Topics")
                        search_input = gr.Textbox(
                            label="Search Query",
                            placeholder="Search topics by name or category..."
                        )
                        search_btn = gr.Button("🔍 Search", variant="secondary")
                        search_output = gr.Markdown()
                
                update_btn.click(
                    interface.update_topic_status_interface,
                    inputs=[topic_name_input, status_dropdown],
                    outputs=[status_output]
                )
                
                search_btn.click(
                    interface.search_topics_interface,
                    inputs=[search_input],
                    outputs=[search_output]
                )
                
                # Real-time search
                search_input.change(
                    interface.search_topics_interface,
                    inputs=[search_input],
                    outputs=[search_output]
                )
            
            # Tab 5: About & Help
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About EulerMCP
                
                EulerMCP is an intelligent learning companion that transforms your educational conversations 
                into interactive knowledge graphs. Built for the Model Context Protocol (MCP), it helps you:
                
                ### ✨ Features
                - **🔍 Smart Analysis**: Extract topics and relationships from conversations using advanced NLP
                - **🕸️ Knowledge Graphs**: Visualize your learning as an interactive network
                - **💡 Personalized Suggestions**: Get AI-powered recommendations for what to learn next
                - **📊 Progress Tracking**: Monitor your learning journey with detailed analytics
                - **🎯 Topic Management**: Organize and track your knowledge across different domains
                
                ### 🚀 Getting Started
                1. **Analyze Conversations**: Paste learning conversations in the first tab
                2. **Explore the Graph**: View your knowledge network in the second tab
                3. **Track Progress**: Check analytics to see your learning patterns
                4. **Manage Topics**: Update topic status and search your knowledge base
                
                ### 🛠️ Technical Details
                - **NLP Engine**: spaCy + sentence-transformers for topic extraction
                - **Graph Engine**: NetworkX for relationship analysis
                - **Visualization**: Plotly for interactive graphs
                - **Database**: SQLite for persistent storage
                - **Interface**: Gradio for web-based interaction
                
                ### 📝 Tips for Best Results
                - Include technical terminology in conversations
                - Ask specific questions about concepts you're learning
                - Discuss relationships between different topics
                - Regular analysis helps build a comprehensive knowledge map
                
                ---
                
                **Built with ❤️ for the learning community**
                
                *Version: 0.1.0 | Framework: Model Context Protocol (MCP)*
                """)
    
    return app


def main():
    """Launch the Gradio interface."""
    app = create_gradio_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )


if __name__ == "__main__":
    main()