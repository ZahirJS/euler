# Conversational phrases that should be filtered out
CONVERSATIONAL_PHRASES = {
    'great question', 'good question', 'the difference', 'key differences', 
    'both interfaces', 'the example', 'this means', 'that means',
    'the problem', 'the solution', 'the answer', 'the question',
    'this way', 'that way', 'the way', 'this thing', 'that thing',
    'the time', 'this time', 'first time', 'next time',
    'the user', 'the assistant', 'the conversation', 'the discussion',
    'i want', 'you can', 'we can', 'they can', 'it can',
    'the code', 'the function', 'the method', 'the class'
}

# Generic terms that are too common to be useful topics
GENERIC_TERMS = {
    # Articles and determiners
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'some', 'any',
    'all', 'every', 'each', 'both', 'either', 'neither', 'no', 'none',
    
    # Conjunctions and prepositions
    'and', 'or', 'but', 'so', 'yet', 'nor', 'for', 'because', 'since',
    'with', 'without', 'from', 'to', 'in', 'on', 'at', 'by', 'of',
    'about', 'under', 'over', 'through', 'between', 'among', 'during',
    'before', 'after', 'while', 'until', 'unless', 'if', 'when', 'where',
    
    # Pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'mine', 'yours', 'hers', 'ours', 'theirs', 'myself', 'yourself',
    'himself', 'herself', 'itself', 'ourselves', 'themselves',
    
    # Common verbs
    'work', 'works', 'working', 'worked', 'use', 'uses', 'using', 'used',
    'make', 'makes', 'making', 'made', 'get', 'gets', 'getting', 'got',
    'have', 'has', 'having', 'had', 'do', 'does', 'doing', 'did', 'done',
    'go', 'goes', 'going', 'went', 'gone', 'come', 'comes', 'coming', 'came',
    'see', 'sees', 'seeing', 'saw', 'seen', 'know', 'knows', 'knowing', 'knew',
    'think', 'thinks', 'thinking', 'thought', 'say', 'says', 'saying', 'said',
    'tell', 'tells', 'telling', 'told', 'give', 'gives', 'giving', 'gave',
    'take', 'takes', 'taking', 'took', 'taken', 'find', 'finds', 'finding', 'found',
    'put', 'puts', 'putting', 'set', 'sets', 'setting', 'run', 'runs', 'running', 'ran',
    'show', 'shows', 'showing', 'showed', 'shown', 'try', 'tries', 'trying', 'tried',
    'keep', 'keeps', 'keeping', 'kept', 'let', 'lets', 'letting', 'call', 'calls',
    'calling', 'called', 'ask', 'asks', 'asking', 'asked', 'need', 'needs',
    'needing', 'needed', 'want', 'wants', 'wanting', 'wanted', 'look', 'looks',
    'looking', 'looked', 'feel', 'feels', 'feeling', 'felt', 'seem', 'seems',
    'seeming', 'seemed', 'become', 'becomes', 'becoming', 'became', 'leave',
    'leaves', 'leaving', 'left', 'turn', 'turns', 'turning', 'turned',
    
    # Modal verbs
    'can', 'could', 'will', 'would', 'should', 'shall', 'may', 'might',
    'must', 'ought', 'dare', 'need', 'used',
    
    # Question words (when used generically)
    'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose', 'whom',
    
    # Numbers and quantifiers
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'first', 'second', 'third', 'last', 'next', 'another', 'other', 'same',
    'different', 'new', 'old', 'more', 'most', 'less', 'least', 'much',
    'many', 'few', 'little', 'big', 'small', 'large', 'long', 'short',
    
    # Generic descriptors
    'good', 'bad', 'great', 'best', 'worst', 'better', 'worse', 'right',
    'wrong', 'true', 'false', 'real', 'fake', 'actual', 'possible',
    'impossible', 'easy', 'hard', 'simple', 'complex', 'basic', 'advanced',
    'important', 'useful', 'helpful', 'necessary', 'available', 'free',
    
    # Time-related generic terms
    'time', 'day', 'week', 'month', 'year', 'hour', 'minute', 'second',
    'today', 'yesterday', 'tomorrow', 'now', 'then', 'soon', 'late',
    'early', 'always', 'never', 'sometimes', 'often', 'usually', 'rarely',
    
    # Generic nouns
    'thing', 'things', 'stuff', 'item', 'items', 'part', 'parts',
    'piece', 'pieces', 'way', 'ways', 'kind', 'type', 'sort', 'form',
    'level', 'point', 'case', 'place', 'area', 'side', 'end', 'start',
    'beginning', 'middle', 'top', 'bottom', 'front', 'back', 'inside',
    'outside', 'above', 'below', 'here', 'there', 'everywhere', 'somewhere',
    'anywhere', 'nowhere'
}

# Technical keywords that remain relevant even in 'general' category
TECHNICAL_KEYWORDS = {
    # Software architecture and design
    'interface', 'abstract', 'inheritance', 'polymorphism', 'encapsulation',
    'composition', 'aggregation', 'coupling', 'cohesion', 'separation',
    'abstraction', 'implementation', 'specification', 'contract', 'api',
    'facade', 'adapter', 'decorator', 'observer', 'singleton', 'factory',
    
    # Data types and structures
    'type', 'generic', 'template', 'typedef', 'struct', 'union', 'enum',
    'array', 'list', 'vector', 'map', 'dictionary', 'hash', 'tree', 'graph',
    'stack', 'queue', 'heap', 'set', 'tuple', 'record', 'object', 'entity',
    
    # Algorithms and complexity
    'algorithm', 'complexity', 'efficiency', 'optimization', 'performance',
    'scalability', 'recursion', 'iteration', 'search', 'sort', 'filter',
    'traverse', 'parse', 'serialize', 'deserialize', 'encode', 'decode',
    
    # Software patterns and practices
    'pattern', 'antipattern', 'best practice', 'convention', 'standard',
    'methodology', 'paradigm', 'approach', 'technique', 'strategy',
    'principle', 'rule', 'guideline', 'recommendation', 'practice',
    
    # System architecture
    'architecture', 'microservices', 'monolith', 'distributed', 'concurrent',
    'parallel', 'asynchronous', 'synchronous', 'event-driven', 'reactive',
    'pipeline', 'workflow', 'orchestration', 'choreography', 'federation',
    
    # Data and storage
    'database', 'schema', 'model', 'entity', 'relation', 'table', 'index',
    'query', 'transaction', 'acid', 'consistency', 'isolation', 'durability',
    'normalization', 'denormalization', 'migration', 'backup', 'replication',
    
    # Code organization
    'structure', 'hierarchy', 'namespace', 'package', 'module', 'library',
    'framework', 'toolkit', 'component', 'widget', 'plugin', 'extension',
    'addon', 'middleware', 'wrapper', 'utility', 'helper', 'service',
    
    # Programming concepts
    'method', 'function', 'procedure', 'subroutine', 'closure', 'lambda',
    'callback', 'promise', 'future', 'async', 'await', 'yield', 'generator',
    'iterator', 'stream', 'pipeline', 'filter', 'map', 'reduce', 'fold',
    
    # Object-oriented programming
    'class', 'instance', 'constructor', 'destructor', 'method', 'property',
    'attribute', 'field', 'member', 'static', 'dynamic', 'virtual', 'override',
    'overload', 'public', 'private', 'protected', 'internal', 'sealed', 'final',
    
    # Network and communication
    'protocol', 'endpoint', 'request', 'response', 'client', 'server',
    'socket', 'connection', 'session', 'handshake', 'timeout', 'retry',
    'circuit breaker', 'load balancer', 'proxy', 'gateway', 'firewall',
    
    # Development lifecycle
    'configuration', 'deployment', 'build', 'compile', 'link', 'package',
    'distribution', 'release', 'version', 'branch', 'merge', 'commit',
    'repository', 'staging', 'production', 'environment', 'pipeline',
    
    # Quality assurance
    'testing', 'unit test', 'integration test', 'end-to-end', 'smoke test',
    'regression', 'coverage', 'assertion', 'mock', 'stub', 'spy', 'fixture',
    'debugging', 'profiling', 'monitoring', 'logging', 'tracing', 'metrics',
    
    # Security and authentication
    'security', 'authentication', 'authorization', 'encryption', 'hashing',
    'certificate', 'token', 'session', 'cookie', 'csrf', 'xss', 'injection',
    'validation', 'sanitization', 'firewall', 'sandbox', 'isolation',
    
    # Performance and optimization
    'optimization', 'performance', 'latency', 'throughput', 'bandwidth',
    'caching', 'memoization', 'lazy loading', 'eager loading', 'prefetch',
    'compression', 'minification', 'bundling', 'chunking', 'streaming',
    
    # Concurrency and parallelism
    'thread', 'process', 'coroutine', 'fiber', 'mutex', 'semaphore',
    'lock', 'atomic', 'volatile', 'race condition', 'deadlock', 'livelock',
    'starvation', 'synchronization', 'coordination', 'barrier',
    
    # Modern development concepts
    'reactive', 'functional', 'declarative', 'imperative', 'immutable',
    'mutable', 'pure function', 'side effect', 'referential transparency',
    'monad', 'functor', 'applicative', 'lens', 'prism', 'isomorphism',
    
    # DevOps and infrastructure
    'container', 'orchestration', 'scaling', 'load balancing', 'service mesh',
    'infrastructure', 'provisioning', 'automation', 'ci/cd', 'pipeline',
    'artifact', 'registry', 'secrets', 'config map', 'health check'
}