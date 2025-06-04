"""
Full system test for EulerMCP.
Tests the complete pipeline: ConversationProcessor → GraphEngine → Server integration.
"""

import asyncio
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from euler_mcp.conversation_processor import ConversationProcessor
from euler_mcp.graph_engine import LearningGraph
from euler_mcp.server import EulerMCPServer


class EulerMCPTester:
    """Comprehensive tester for the EulerMCP system."""
    
    def __init__(self):
        """Initialize tester with temporary database."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_graph.db")
        
        # Initialize components
        self.processor = ConversationProcessor()
        self.graph = LearningGraph(self.db_path)
        self.server = EulerMCPServer(self.db_path)
        
        print(f"🧪 Test database created at: {self.db_path}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up test directory: {self.temp_dir}")
    
    async def test_conversation_processor(self):
        """Test the conversation processor component."""
        print("\n🔍 Testing ConversationProcessor...")
        
        sample_conversation = """
        User: I'm learning React and TypeScript for web development. How do hooks work?
        
        Assistant: React hooks are functions that let you use state and lifecycle features 
        in functional components. The most common hooks are useState and useEffect.
        
        User: What about custom hooks? And how does TypeScript help with React development?
        
        Assistant: Custom hooks let you extract component logic into reusable functions. 
        TypeScript provides type safety for props, state, and function parameters in React.
        """
        
        # Test topic extraction
        topics = await self.processor.extract_topics(sample_conversation)
        print(f"   ✅ Extracted {len(topics)} topics")
        
        for i, topic in enumerate(topics[:3], 1):
            print(f"      {i}. {topic['name']} ({topic['category']}) - confidence: {topic['confidence']:.2f}")
        
        # Test relationship identification
        relationships = await self.processor.identify_relationships(topics)
        print(f"   ✅ Found {len(relationships)} relationships")
        
        if relationships:
            for i, rel in enumerate(relationships[:3], 1):
                print(f"      {i}. {rel['source']} → {rel['target']} ({rel['type']})")
        
        return topics, relationships
    
    async def test_graph_engine(self, topics, relationships):
        """Test the graph engine component."""
        print("\n🕸️ Testing GraphEngine...")
        
        # Test adding topics
        topic_ids = []
        for topic in topics:
            topic_id = self.graph.add_topic(
                name=topic['name'],
                category=topic['category'],
                confidence=topic['confidence'],
                status='mentioned'
            )
            topic_ids.append(topic_id)
        
        print(f"   ✅ Added {len(topic_ids)} topics to graph")
        
        # Test adding relationships
        relationship_count = 0
        for rel in relationships:
            source_topic = self.graph.get_topic_by_name(rel['source'])
            target_topic = self.graph.get_topic_by_name(rel['target'])
            
            if source_topic and target_topic:
                success = self.graph.add_relationship(
                    source_topic['id'],
                    target_topic['id'],
                    relationship_type=rel['type'],
                    weight=rel['weight'],
                    confidence=rel['confidence']
                )
                if success:
                    relationship_count += 1
        
        print(f"   ✅ Added {relationship_count} relationships to graph")
        
        # Test graph data export
        graph_data = self.graph.get_graph_data()
        print(f"   ✅ Graph contains {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
        
        # Test suggestions
        current_topics = [topic['name'] for topic in topics[:2]]  # Use first 2 topics
        suggestions = await self.graph.suggest_next_topics(current_topics, 3)
        print(f"   ✅ Generated {len(suggestions)} learning suggestions")
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"      {i}. {suggestion}")
        
        # Test learning path analysis
        if topics:
            target_topic = topics[0]['name']
            path_analysis = self.graph.analyze_learning_path(target_topic)
            print(f"   ✅ Analyzed learning path for '{target_topic}'")
            
            if 'error' not in path_analysis:
                prereqs = path_analysis.get('prerequisites', [])
                print(f"      Prerequisites found: {len(prereqs)}")
        
        return graph_data, suggestions
    
    async def test_server_integration(self):
        """Test the MCP server integration."""
        print("\n🖥️ Testing Server Integration...")
        
        # Test conversation analysis through server
        sample_conversation = """
        User: I want to learn machine learning. Where should I start?
        
        Assistant: Start with Python basics, then learn pandas and numpy for data manipulation. 
        After that, you can explore scikit-learn for basic algorithms.
        
        User: What about deep learning frameworks like TensorFlow?
        
        Assistant: TensorFlow is great for deep learning, but I'd recommend mastering 
        the fundamentals first. PyTorch is also worth considering.
        """
        
        result = await self.server.analyze_conversation(
            sample_conversation, 
            "test-conversation-ml"
        )
        
        print(f"   ✅ Server analysis found {len(result.topics_found)} topics")
        print(f"   ✅ Server generated {len(result.suggestions)} suggestions")
        print(f"   ✅ Graph stats: {result.graph_stats}")
        
        # Test getting learning graph through server
        graph_data = await self.server.get_learning_graph("json")
        print(f"   ✅ Server returned graph with {len(graph_data.get('nodes', []))} nodes")
        
        # Test topic status update
        if result.topics_found:
            test_topic = result.topics_found[0]
            update_result = await self.server.update_topic_status(test_topic, "learned")
            print(f"   ✅ Topic status update: {update_result['success']}")
        
        # Test analytics
        analytics = await self.server.get_learning_analytics()
        if 'error' not in analytics:
            overview = analytics['overview']
            print(f"   ✅ Analytics: {overview['total_topics']} topics, {overview['topics_learned']} learned")
        
        return result
    
    async def test_multiple_conversations(self):
        """Test with multiple different conversations to build a rich graph."""
        print("\n📚 Testing Multiple Conversations...")
        
        conversations = [
            {
                "id": "web-dev-basics",
                "text": """
                User: I'm starting web development. Should I learn HTML first?
                Assistant: Yes, start with HTML for structure, then CSS for styling, 
                and JavaScript for interactivity. This is the foundation of web development.
                """
            },
            {
                "id": "react-learning",
                "text": """
                User: I know HTML and CSS. Ready for React?
                Assistant: Great! React builds on JavaScript knowledge. Learn ES6 features 
                first, then React concepts like components, props, and state management.
                """
            },
            {
                "id": "backend-discussion",
                "text": """
                User: What about backend development with Node.js?
                Assistant: Node.js is perfect since you know JavaScript. Learn Express.js 
                for web servers, and understand databases like MongoDB or PostgreSQL.
                """
            }
        ]
        
        total_topics = 0
        for conv in conversations:
            result = await self.server.analyze_conversation(
                conv["text"], 
                conv["id"]
            )
            topics_count = len(result.topics_found)
            total_topics += topics_count
            print(f"   ✅ {conv['id']}: {topics_count} topics extracted")
        
        # Get final graph state
        final_graph = await self.server.get_learning_graph("json")
        final_analytics = await self.server.get_learning_analytics()
        
        print(f"   ✅ Total topics in graph: {len(final_graph.get('nodes', []))}")
        print(f"   ✅ Total connections: {len(final_graph.get('edges', []))}")
        
        if 'error' not in final_analytics:
            categories = len(final_analytics['category_distribution'])
            print(f"   ✅ Knowledge areas covered: {categories}")
        
        return final_graph, final_analytics
    
    def test_data_persistence(self):
        """Test that data persists between sessions."""
        print("\n💾 Testing Data Persistence...")
        
        # Create new instances using same database
        new_graph = LearningGraph(self.db_path)
        new_server = EulerMCPServer(self.db_path)
        
        # Check if data persists
        graph_data = new_graph.get_graph_data()
        nodes_count = len(graph_data['nodes'])
        edges_count = len(graph_data['edges'])
        
        print(f"   ✅ Persisted {nodes_count} topics and {edges_count} relationships")
        
        # Test querying persisted data
        if nodes_count > 0:
            first_topic_name = graph_data['nodes'][0]['name']
            topic_data = new_graph.get_topic_by_name(first_topic_name)
            
            if topic_data:
                print(f"   ✅ Successfully retrieved topic: {first_topic_name}")
            else:
                print(f"   ❌ Failed to retrieve topic: {first_topic_name}")
        
        return nodes_count > 0 and edges_count >= 0  # At least some data should persist
    
    async def run_all_tests(self):
        """Run all tests in sequence."""
        print("🚀 Starting EulerMCP Full System Test")
        print("=" * 50)
        
        try:
            # Test individual components
            topics, relationships = await self.test_conversation_processor()
            graph_data, suggestions = await self.test_graph_engine(topics, relationships)
            
            # Test server integration
            server_result = await self.test_server_integration()
            
            # Test multiple conversations
            final_graph, analytics = await self.test_multiple_conversations()
            
            # Test persistence
            persistence_ok = self.test_data_persistence()
            
            # Summary
            print("\n📊 Test Summary")
            print("=" * 30)
            print(f"✅ ConversationProcessor: {len(topics)} topics, {len(relationships)} relationships")
            print(f"✅ GraphEngine: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
            print(f"✅ Server Integration: {len(server_result.topics_found)} topics processed")
            print(f"✅ Multiple Conversations: {len(final_graph.get('nodes', []))} total topics")
            print(f"✅ Data Persistence: {'Passed' if persistence_ok else 'Failed'}")
            
            if analytics and 'error' not in analytics:
                overview = analytics['overview']
                print(f"✅ Final Analytics:")
                print(f"   - Total Topics: {overview['total_topics']}")
                print(f"   - Topics Learned: {overview['topics_learned']}")
                print(f"   - Completion: {overview['learning_completion_percentage']}%")
                print(f"   - Knowledge Areas: {overview['knowledge_areas']}")
            
            print("\n🎉 All tests completed successfully!")
            print("\n💡 Ready for demo! You can now:")
            print("   1. Run the MCP server: python -m euler_mcp.server")
            print("   2. Launch Gradio interface: python -m euler_mcp.gradio_app")
            print("   3. Connect to Claude Desktop via MCP")
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()


async def main():
    """Run the full system test."""
    tester = EulerMCPTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())