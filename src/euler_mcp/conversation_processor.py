"""

ConversationProcessor: Extract topics and relations of conversations using NLP.

This module is responsible of:
1. Processing text of conversations.
2. Extracting relevants topics/concepts 
3. Identifying relations between topics.
4. Classifying learning deepness.

"""

import re 
import spacy 
from typing import List, Dict, Any, Set
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .config import LEARNING_PATTERNS, TOPIC_CATEGORIES, CONVERSATIONAL_PHRASES, GENERIC_TERMS, TECHNICAL_KEYWORDS, PREREQUISITES
class ConversationProcessor:
    """Principal processor for extracting knowledge from conversations."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the processor with NPL models.
        
        Args: 
            model_name (str): Name of the spaCy model to use for NLP tasks.
        """

        # Load spaCy model for sintatic analysis and entities
        self.nlp = spacy.load(model_name)

        # Load embedding model for semantic analysis
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        #Patterns to detect different types of learning
        self.learning_patterns = LEARNING_PATTERNS
        #Categories of common technical topics
        self.topic_categories = TOPIC_CATEGORIES

    async def extract_topics(self, conversation_text: str) -> List[Dict[str, Any]]:
        """
        Extract main topics from a conversation.
        
        Args: 
            conversation_text (str): Full conversation text.
            
        Returns:
            List of dictionaries with inforamation about each topic.
        """

        doc = self.nlp(conversation_text)

        entities = self._extract_entities(doc)
        technical_terms = self._extract_technical_terms(doc)
        concepts = self._extract_concepts(doc)

        # Combine and deduplicate all found topics
        all_topics = entities + technical_terms + concepts
        unique_topics = self._deduplicate_topics(all_topics)

        # Enrich each topic with additional metadata
        enriched_topics = []
        for topic in unique_topics:
            enriched_topic = await self._enrich_topic(topic, conversation_text)
            enriched_topics.append(enriched_topic)

        # Filter low-relevance topics and sort by relevance
        filtered_topics = self._filter_relevant_topics(enriched_topics)

        return sorted(filtered_topics, key=lambda x: x['confidence'], reverse=True)
    
    def _extract_entities(self, doc) -> List[str]:
        """
        Extract named entities from text using spaCy.
        
        Args:
            doc: Document processed by spaCy.
        
        Returns:
            List of relevant entities.
        """

        entities = []

        relevant_labels = {'PERSON', 'ORG', 'PRODUCT', 'EVENT', 'LANGUAGE', 'TECH'}

        for ent in doc.ents:
            if len(ent.text) > 2 and ent.label_ in relevant_labels:
                entities.append(ent.text.lower().strip())
        
        return entities

    def _extract_technical_terms(self, doc) -> List[str]:
        """
        Extract technical terms using patterns and specialized vocabulary.
        
        Args:
            doc: Document processed by spaCy
            
        Returns:
            List of technical terms found
        """
        technical_terms = []
        text_lower = doc.text.lower()
        
        # Search for technical terms in all categories using word boundaries
        for category, terms in self.topic_categories.items():
            for term in terms:
                # Use regex with word boundaries to match complete words/phrases only
                import re
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, text_lower):
                    technical_terms.append(term)
        
        # Search for additional patterns (camelCase, snake_case, etc.)
        camel_case_pattern = r'\b[a-z]+(?:[A-Z][a-z]+)+\b'
        snake_case_pattern = r'\b[a-z]+(?:_[a-z]+)+\b'
        
        camel_matches = re.findall(camel_case_pattern, doc.text)
        snake_matches = re.findall(snake_case_pattern, doc.text)
        
        technical_terms.extend([match.lower() for match in camel_matches])
        technical_terms.extend([match.lower() for match in snake_matches])
        
        return technical_terms

    def _extract_concepts(self, doc) -> List[str]:
        """
        Extract concepts using syntactic analysis (important noun phrases).
        
        Args:
            doc: Document processed by spaCy
            
        Returns:
            List of extracted concepts
        """
        concepts = []

        for chunk in doc.noun_chunks:
            # Filter out short phrases and common words
            if(len(chunk.text.split()) >= 2 and len(chunk.text) > 4 and not self._is_common_phrase(chunk.text)):
                concepts.append(chunk.text.lower().strip())

        return concepts

    def _is_common_phrase(self, phrase: str) -> bool:
        """
        Determine if a phrase is too common to be relevant.
        
        Args:
            phrase: Phrase to evaluate
            
        Returns:
            True if the phrase is very common
        """
        common_phrases = {
            'the way', 'this thing', 'that thing', 'the time', 'the problem',
            'the question', 'the answer', 'the example', 'the code', 'the function'
        }
        return phrase.lower() in common_phrases
    

    def _deduplicate_topics(self, topics: List[str]) -> List[str]:
        """
        Remove duplicate or very similar topics.
        
        Args:
            topics: List of topics with possible duplicates
            
        Returns:
            List of unique topics
        """

        if not topics:
            return []
        
        normalized = list(set([topic.lower().strip() for topic in topics if len(topic.strip()) > 2]))

        # Use embeddings to find very similar topics
        if len(normalized) > 1:
            embeddings = self.sentence_model.encode(normalized)
            similarity_matrix = cosine_similarity(embeddings)
            
            # Remove topics with similarity > 0.8
            unique_indices = []
            for i, topic in enumerate(normalized):
                is_unique = True
                for j in unique_indices:
                    if similarity_matrix[i][j] > 0.8:
                        is_unique = False
                        break
                if is_unique:
                    unique_indices.append(i)
            
            return [normalized[i] for i in unique_indices]
        
        return normalized
    
    async def _enrich_topic(self, topic: str, full_text: str) -> Dict[str, Any]:
        """
        Enrich a topic with additional metadata.
        
        Args:
            topic: Topic name
            full_text: Full text for context analysis
            
        Returns:
            Dictionary with enriched topic information
        """

        # Determine topic category
        category = self._categorize_topic(topic)
        
        # Calculate confidence based on frequency and context
        confidence = self._calculate_confidence(topic, full_text)
        
        # Determine learning depth
        depth = self._analyze_depth(topic, full_text)
        
        # Extract relevant context
        context = self._extract_context(topic, full_text)
        
        return {
            'name': topic,
            'category': category,
            'confidence': confidence,
            'depth': depth,
            'context': context,
            'frequency': full_text.lower().count(topic.lower())
        }

    def _categorize_topic(self, topic: str) -> str:
        """
        Categorize a topic based on its content.
        
        Args:
            topic: Topic name
            
        Returns:
            Topic category
        """
        topic_lower = topic.lower()
        
        for category, terms in self.topic_categories.items():
            if any(term in topic_lower for term in terms):
                return category
        
        return 'general'

    def _calculate_confidence(self, topic: str, full_text: str) -> float:
        """
        Calculate confidence level for a topic based on context.
        
        Args:
            topic: Topic name
            full_text: Full text
            
        Returns:
            Confidence value between 0 and 1
        """
        topic_lower = topic.lower()
        text_lower = full_text.lower()
        
        # Base factors
        frequency = text_lower.count(topic_lower)
        length_bonus = min(len(topic.split()) * 0.1, 0.3)  # Multi-word topics are more specific
        
        # Learning context bonus
        learning_context = 0
        for pattern_type, patterns in self.learning_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    learning_context += 0.15
                    break
        
        # Technical relevance bonus
        category = self._categorize_topic(topic)
        technical_bonus = 0.3 if self._is_technical_category(category) else 0
        
        # Penalty for conversational phrases
        conversational_penalty = -0.4 if self._is_conversational_phrase(topic) else 0
        
        # Penalty for very generic terms
        generic_penalty = -0.2 if self._is_too_generic(topic) else 0
        
        # Calculate base confidence from frequency
        base_confidence = min(frequency * 0.15, 0.6)
        
        # Calculate final confidence
        total_confidence = (base_confidence + 
                        length_bonus + 
                        learning_context + 
                        technical_bonus + 
                        conversational_penalty + 
                        generic_penalty)
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(round(total_confidence, 2), 1.0))

    def _analyze_depth(self, topic: str, full_text: str) -> str:
        """
        Analyze the learning depth of a topic.
        
        Args:
            topic: Topic name
            full_text: Full text
            
        Returns:
            Depth level: 'shallow', 'medium', 'deep'
        """
        topic_lower = topic.lower()
        text_lower = full_text.lower()
        
        # Search for depth indicators
        shallow_indicators = ['what is', 'qué es', 'introduction', 'basic', 'overview']
        medium_indicators = ['how to', 'cómo', 'example', 'practice', 'implement']
        deep_indicators = ['advanced', 'optimization', 'best practices', 'architecture', 'design patterns']
        
        # Count indicators near the topic
        context_window = self._get_context_around_topic(topic, full_text, 100)
        
        deep_count = sum(1 for indicator in deep_indicators if indicator in context_window)
        medium_count = sum(1 for indicator in medium_indicators if indicator in context_window)
        shallow_count = sum(1 for indicator in shallow_indicators if indicator in context_window)
        
        if deep_count > 0:
            return 'deep'
        elif medium_count > shallow_count:
            return 'medium'
        else:
            return 'shallow'

    def _extract_context(self, topic: str, full_text: str) -> str:
        """
        Extract relevant context around a topic.
        
        Args:
            topic: Topic name
            full_text: Full text
            
        Returns:
            Extracted topic context
        """
        return self._get_context_around_topic(topic, full_text, 200)

    def _get_context_around_topic(self, topic: str, text: str, window_size: int) -> str:
        """
        Get context around the first mention of a topic.
        
        Args:
            topic: Topic to search for
            text: Text to search in
            window_size: Context window size
            
        Returns:
            Extracted context
        """
        topic_lower = topic.lower()
        text_lower = text.lower()
        
        index = text_lower.find(topic_lower)
        if index == -1:
            return ""
        
        start = max(0, index - window_size)
        end = min(len(text), index + len(topic) + window_size)
        
        return text[start:end].strip()

    def _filter_relevant_topics(self, topics: List[Dict[str, Any]], min_confidence: float = 0.2) -> List[Dict[str, Any]]:
        """
        Filter low-relevance topics based on confidence and other criteria.
        
        Args:
            topics: List of enriched topics
            min_confidence: Minimum required confidence
            
        Returns:
            Filtered list of relevant topics
        """
        filtered_topics = []
        
        for topic in topics:
            # Basic filters
            if (len(topic['name']) <= 2 or 
                topic['name'].isdigit() or
                topic['confidence'] < min_confidence):
                continue
                
            # Filter conversational phrases
            if self._is_conversational_phrase(topic['name']):
                continue
                
            # Filter very common words
            if self._is_too_generic(topic['name']):
                continue
                
            # Prefer technical topics over general ones
            if (topic['category'] == 'general' and 
                topic['confidence'] < 0.4 and 
                not self._is_technical_relevant(topic['name'])):
                continue
                
            filtered_topics.append(topic)
        
        return filtered_topics

    def _is_conversational_phrase(self, phrase: str) -> bool:
        """
        Determine if a phrase is conversational rather than a technical concept.
        
        Args:
            phrase: Phrase to evaluate
            
        Returns:
            True if the phrase is conversational/non-technical
        """
        return phrase.lower().strip() in CONVERSATIONAL_PHRASES

    def _is_too_generic(self, topic: str) -> bool:
        """
        Check if a topic is too generic to be useful.
        
        Args:
            topic: Topic name
            
        Returns:
            True if topic is too generic
        """
        return topic.lower().strip() in GENERIC_TERMS

    def _is_technical_relevant(self, topic: str) -> bool:
        """
        Check if a general topic is still technically relevant.
        
        Args:
            topic: Topic name
            
        Returns:
            True if topic is technically relevant despite being 'general' category
        """
        topic_words = set(topic.lower().split())
        return bool(topic_words.intersection(TECHNICAL_KEYWORDS))

    def _is_technical_category(self, category: str) -> bool:
        """
        Determine if a category is considered technical (not general).
        
        Args:
            category: Category name
            
        Returns:
            True if category is technical
        """
        return category != 'general'

    async def identify_relationships(self, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify relationships between extracted topics.
        
        Args:
            topics: List of enriched topics
            
        Returns:
            List of relationships between topics
        """
        relationships = []
        
        if len(topics) < 2:
            return relationships
        
        # Get embeddings for all topics
        topic_names = [topic['name'] for topic in topics]
        embeddings = self.sentence_model.encode(topic_names)
        
        # Calculate similarities between all pairs
        for i, topic_a in enumerate(topics):
            for j, topic_b in enumerate(topics[i+1:], i+1):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                
                if similarity > 0.3:  # Similarity threshold
                    relationship_type = self._determine_relationship_type(
                        topic_a, topic_b, similarity
                    )
                    
                    relationships.append({
                        'source': topic_a['name'],
                        'target': topic_b['name'],
                        'type': relationship_type,
                        'weight': round(similarity, 3),
                        'confidence': min(topic_a['confidence'], topic_b['confidence'])
                    })
        
        return relationships

    def _determine_relationship_type(self, topic_a: Dict[str, Any], topic_b: Dict[str, Any], similarity: float) -> str:
        """
        Determine the type of relationship between two topics.
        
        Args:
            topic_a: First topic
            topic_b: Second topic
            similarity: Semantic similarity
            
        Returns:
            Relationship type
        """
        # If they're from the same category and very similar
        if (topic_a['category'] == topic_b['category'] and similarity > 0.7):
            return 'related'
        
        # If one seems to be a prerequisite of the other
        if self._is_prerequisite(topic_a['name'], topic_b['name']):
            return 'prerequisite'
        
        # If one extends the other
        if self._is_extension(topic_a['name'], topic_b['name']):
            return 'extension'
        
        # Default general relationship
        return 'related'

    def _is_prerequisite(self, topic_a: str, topic_b: str) -> bool:
        """
        Determine if a topic is a prerequisite of another.
        
        Args:
            topic_a: First topic
            topic_b: Second topic
            
        Returns:
            True if topic_a is a prerequisite of topic_b
        """
        # Define some known prerequisites
        prerequisites = PREREQUISITES
        
        topic_a_lower = topic_a.lower()
        topic_b_lower = topic_b.lower()
        
        return topic_b_lower in prerequisites.get(topic_a_lower, [])

    def _is_extension(self, topic_a: str, topic_b: str) -> bool:
        """
        Determine if a topic is an extension of another.
        
        Args:
            topic_a: First topic
            topic_b: Second topic
            
        Returns:
            True if topic_b extends topic_a
        """
        # If one topic contains the other, it could be an extension
        return (topic_a.lower() in topic_b.lower() and 
                topic_a.lower() != topic_b.lower() and
                len(topic_b.split()) > len(topic_a.split()))