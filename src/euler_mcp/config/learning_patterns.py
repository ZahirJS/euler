LEARNING_PATTERNS = {
    'question': [
        # Basic question words
        r'\b(?:what|how|why|when|where|who|which|whose)\b.*\?',
        r'explain.*',
        r'tell me.*',
        r'can you.*',
        r'could you.*',
        r'would you.*',
        r'will you.*',
        
        # Direct questions
        r'.*\?$',
        r'describe.*',
        r'define.*',
        r'show me.*',
        r'walk me through.*',
        r'help me understand.*',
        r'clarify.*',
        r'elaborate.*',
        
        # Question starters
        r'^(?:is|are|do|does|did|can|could|will|would|should|might|may).*\?',
        r'i want to know.*',
        r'i need to know.*',
        r'i\'d like to know.*',
        r'what\'s the difference.*',
        r'what are the.*',
        r'how do.*',
        r'how can.*',
        r'how would.*',
        r'why is.*',
        r'why does.*',
        r'when should.*',
        r'where can.*',
        r'which one.*',
        
        # Tutorial requests
        r'tutorial.*',
        r'guide.*',
        r'step by step.*',
        r'walkthrough.*',
        r'how-to.*',
        r'instructions.*',
        r'teach me.*',
        r'show me how.*'
    ],
    
    'confusion': [
        # Direct confusion expressions
        r'\b(?:confused|don\'t understand|unclear|not sure|uncertain|puzzled|lost|stuck)\b',
        r'\b(?:confusing|complicated|complex|difficult|hard to understand)\b',
        r'\b(?:i don\'t get|doesn\'t make sense|not clear|ambiguous)\b',
        
        # Seeking clarification
        r'can you clarify.*',
        r'i\'m not following.*',
        r'i\'m having trouble.*',
        r'i\'m struggling.*',
        r'this is confusing.*',
        r'i\'m lost.*',
        r'this doesn\'t make sense.*',
        r'i don\'t see how.*',
        r'i\'m not sure what.*',
        r'i\'m not clear on.*',
        
        # Question marks indicating confusion
        r'huh\?',
        r'what\?',
        r'sorry\?',
        r'come again\?',
        r'could you repeat.*',
        r'say that again.*',
        
        # Uncertainty expressions
        r'i think but.*',
        r'maybe but.*',
        r'not quite sure.*',
        r'kind of confused.*',
        r'a bit lost.*',
        r'somewhat unclear.*'
    ],
    
    'learning': [
        # Learning activities
        r'\b(?:learning|studying|practice|practicing|understand|understanding|mastering)\b',
        r'\b(?:student|beginner|newbie|novice|learner|apprentice)\b',
        r'\b(?:course|class|lesson|training|education|tutorial|bootcamp)\b',
        r'\b(?:homework|assignment|project|exercise|quiz|exam|test)\b',
        
        # Learning intentions
        r'want to learn.*',
        r'trying to learn.*',
        r'need to learn.*',
        r'learning about.*',
        r'studying.*',
        r'working on.*',
        r'practicing.*',
        r'getting into.*',
        r'diving into.*',
        r'exploring.*',
        r'familiarizing.*',
        
        # Skill development
        r'improve my.*',
        r'get better at.*',
        r'develop.*skills',
        r'build.*knowledge',
        r'expand my.*',
        r'strengthen my.*',
        r'enhance my.*',
        r'master.*',
        r'become proficient.*',
        r'gain experience.*',
        
        # Learning context
        r'i\'m new to.*',
        r'just started.*',
        r'beginning with.*',
        r'first time.*',
        r'recently started.*',
        r'getting started.*',
        r'starting out.*',
        r'picking up.*',
        r'self-taught.*',
        r'autodidact.*',
        
        # Educational levels
        r'undergraduate.*',
        r'graduate.*',
        r'phd.*',
        r'doctorate.*',
        r'college.*',
        r'university.*',
        r'high school.*',
        r'bootcamp.*',
        r'certification.*',
        r'degree.*'
    ],
    
    'problem_solving': [
        # Problem identification
        r'problem.*',
        r'issue.*',
        r'error.*',
        r'bug.*',
        r'trouble.*',
        r'difficulty.*',
        r'challenge.*',
        r'obstacle.*',
        r'roadblock.*',
        
        # Help seeking
        r'help.*',
        r'assist.*',
        r'support.*',
        r'solve.*',
        r'fix.*',
        r'resolve.*',
        r'troubleshoot.*',
        r'debug.*',
        
        # Specific problem patterns
        r'not working.*',
        r'doesn\'t work.*',
        r'broken.*',
        r'failing.*',
        r'getting an error.*',
        r'keeps crashing.*',
        r'won\'t start.*',
        r'can\'t get.*to work',
        r'stuck on.*',
        r'having issues with.*'
    ],
    
    'comparative_learning': [
        # Comparisons and choices
        r'difference between.*',
        r'compare.*',
        r'versus.*',
        r'vs\..*',
        r'better.*',
        r'best.*',
        r'worst.*',
        r'pros and cons.*',
        r'advantages.*',
        r'disadvantages.*',
        r'which is.*',
        r'which should.*',
        r'what\'s better.*',
        r'should i use.*',
        r'or.*\?',
        
        # Decision making
        r'choose between.*',
        r'pick.*',
        r'select.*',
        r'decide.*',
        r'recommend.*',
        r'suggest.*',
        r'advice.*',
        r'opinion.*',
        r'thoughts on.*'
    ],
    
    'hands_on_learning': [
        # Practical application
        r'example.*',
        r'demo.*',
        r'demonstration.*',
        r'sample.*',
        r'code.*',
        r'implementation.*',
        r'build.*',
        r'create.*',
        r'make.*',
        r'develop.*',
        
        # Practice requests
        r'practice.*',
        r'exercise.*',
        r'try.*',
        r'test.*',
        r'experiment.*',
        r'play around.*',
        r'hands-on.*',
        r'practical.*',
        r'real-world.*',
        r'use case.*',
        
        # Project-based
        r'project.*',
        r'assignment.*',
        r'task.*',
        r'challenge.*',
        r'portfolio.*',
        r'side project.*',
        r'pet project.*'
    ],
    
    'conceptual_learning': [
        # Understanding concepts
        r'concept.*',
        r'theory.*',
        r'principle.*',
        r'fundamentals.*',
        r'basics.*',
        r'foundation.*',
        r'core.*',
        r'essence.*',
        r'meaning.*',
        r'definition.*',
        
        # Deep learning
        r'deep dive.*',
        r'in-depth.*',
        r'comprehensive.*',
        r'thorough.*',
        r'detailed.*',
        r'complete.*',
        r'full.*',
        r'everything about.*',
        r'all about.*',
        
        # Mental models
        r'mental model.*',
        r'framework.*',
        r'paradigm.*',
        r'approach.*',
        r'methodology.*',
        r'strategy.*',
        r'pattern.*',
        r'architecture.*'
    ],
    
    'progress_tracking': [
        # Learning progress
        r'progress.*',
        r'improvement.*',
        r'growth.*',
        r'development.*',
        r'advancement.*',
        r'milestone.*',
        r'checkpoint.*',
        r'level.*',
        
        # Self-assessment
        r'how am i doing.*',
        r'am i on track.*',
        r'feedback.*',
        r'review.*',
        r'assessment.*',
        r'evaluation.*',
        r'check my.*',
        r'validate my.*',
        
        # Next steps
        r'what\'s next.*',
        r'next step.*',
        r'where to go.*',
        r'what should i.*',
        r'roadmap.*',
        r'path.*',
        r'journey.*',
        r'continue.*'
    ]
}