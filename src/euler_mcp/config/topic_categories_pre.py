PREREQUISITES = {
    # Programming Languages - Foundational
    'html': ['text editors', 'basic computer skills'],
    'css': ['html'],
    'javascript': ['html', 'css'],
    'typescript': ['javascript'],
    'python': ['basic programming concepts'],
    'java': ['basic programming concepts', 'object oriented programming'],
    'c++': ['c', 'object oriented programming'],
    'c': ['basic programming concepts'],
    'rust': ['c', 'memory management'],
    'go': ['c', 'basic programming concepts'],
    
    # Web Development - Frontend
    'react': ['javascript', 'html', 'css'],
    'vue': ['javascript', 'html', 'css'],
    'angular': ['typescript', 'javascript', 'html', 'css'],
    'svelte': ['javascript', 'html', 'css'],
    'nextjs': ['react', 'javascript'],
    'nuxt': ['vue', 'javascript'],
    'gatsby': ['react', 'graphql'],
    'tailwind css': ['css', 'html'],
    'bootstrap': ['css', 'html'],
    'sass': ['css'],
    'less': ['css'],
    
    # Web Development - Backend
    'node.js': ['javascript'],
    'express': ['node.js', 'javascript'],
    'nestjs': ['typescript', 'node.js'],
    'django': ['python'],
    'flask': ['python'],
    'fastapi': ['python'],
    'spring': ['java'],
    'spring boot': ['spring', 'java'],
    'rails': ['ruby'],
    'laravel': ['php'],
    'asp.net': ['c#'],
    
    # Mobile Development
    'react native': ['react', 'javascript'],
    'flutter': ['dart'],
    'android': ['java', 'kotlin'],
    'ios': ['swift', 'objective-c'],
    'xamarin': ['c#'],
    'ionic': ['angular', 'html', 'css'],
    
    # Data Science & ML
    'pandas': ['python', 'numpy'],
    'numpy': ['python'],
    'matplotlib': ['python', 'numpy'],
    'seaborn': ['matplotlib', 'pandas'],
    'scikit-learn': ['python', 'numpy', 'pandas'],
    'tensorflow': ['python', 'numpy', 'machine learning'],
    'pytorch': ['python', 'numpy', 'machine learning'],
    'keras': ['tensorflow', 'python'],
    'machine learning': ['statistics', 'linear algebra', 'python'],
    'deep learning': ['machine learning', 'neural networks'],
    'neural networks': ['machine learning', 'linear algebra'],
    'computer vision': ['python', 'numpy', 'machine learning'],
    'nlp': ['python', 'machine learning', 'statistics'],
    
    # Database Systems
    'postgresql': ['sql', 'database design'],
    'mysql': ['sql', 'database design'],
    'mongodb': ['database design', 'json'],
    'redis': ['key-value concepts'],
    'elasticsearch': ['json', 'search concepts'],
    'sql': ['database design'],
    'database design': ['data modeling'],
    
    # Cloud & DevOps
    'docker': ['linux', 'virtualization'],
    'kubernetes': ['docker', 'containers', 'networking'],
    'aws': ['cloud computing concepts'],
    'azure': ['cloud computing concepts'],
    'gcp': ['cloud computing concepts'],
    'terraform': ['infrastructure as code', 'cloud computing'],
    'ansible': ['linux', 'ssh', 'yaml'],
    'jenkins': ['ci/cd concepts'],
    'github actions': ['git', 'yaml', 'ci/cd concepts'],
    'prometheus': ['monitoring concepts', 'yaml'],
    'grafana': ['prometheus', 'monitoring concepts'],
    
    # Version Control & Collaboration
    'git': ['command line'],
    'github': ['git'],
    'gitlab': ['git'],
    'version control': ['basic programming concepts'],
    
    # Operating Systems & System Administration
    'linux': ['command line', 'operating systems'],
    'bash': ['linux', 'command line'],
    'zsh': ['bash', 'shell scripting'],
    'powershell': ['windows', 'command line'],
    'ubuntu': ['linux'],
    'centos': ['linux'],
    'docker': ['linux', 'virtualization'],
    
    # Networking & Security
    'tcp/ip': ['networking fundamentals'],
    'http': ['tcp/ip', 'networking'],
    'https': ['http', 'ssl', 'tls'],
    'dns': ['networking fundamentals'],
    'ssl': ['cryptography', 'networking'],
    'tls': ['ssl', 'cryptography'],
    'vpn': ['networking', 'security'],
    'firewall': ['networking', 'security'],
    
    # Package Managers
    'npm': ['node.js', 'javascript'],
    'yarn': ['npm', 'node.js'],
    'pnpm': ['npm', 'node.js'],
    'pip': ['python'],
    'poetry': ['pip', 'python'],
    'conda': ['python'],
    'cargo': ['rust'],
    'maven': ['java'],
    'gradle': ['java'],
    
    # Testing
    'jest': ['javascript', 'testing concepts'],
    'cypress': ['javascript', 'testing concepts'],
    'selenium': ['programming', 'web development'],
    'pytest': ['python', 'testing concepts'],
    'junit': ['java', 'testing concepts'],
    'testing concepts': ['programming fundamentals'],
    
    # Build Tools
    'webpack': ['javascript', 'node.js'],
    'vite': ['javascript', 'node.js'],
    'babel': ['javascript', 'es6'],
    'typescript compiler': ['typescript'],
    'rollup': ['javascript', 'modules'],
    
    # Advanced Concepts
    'microservices': ['api design', 'distributed systems'],
    'graphql': ['api design', 'databases'],
    'rest api': ['http', 'api design'],
    'websockets': ['http', 'networking'],
    'oauth': ['authentication', 'security'],
    'jwt': ['authentication', 'json'],
    'docker compose': ['docker', 'yaml'],
    'kubernetes': ['docker', 'orchestration'],
    
    # LLM & AI Models (2025)
    'gpt-4o': ['nlp', 'transformer models'],
    'claude 4': ['nlp', 'ai fundamentals'],
    'gemini': ['nlp', 'machine learning'],
    'llama 4': ['nlp', 'transformer models'],
    'transformer models': ['deep learning', 'attention mechanism'],
    'attention mechanism': ['neural networks', 'deep learning'],
    'model context protocol': ['ai fundamentals', 'api design'],
    
    # Blockchain & Crypto
    'smart contracts': ['blockchain', 'solidity'],
    'solidity': ['programming fundamentals', 'blockchain'],
    'ethereum': ['blockchain', 'cryptography'],
    'web3': ['blockchain', 'javascript'],
    'defi': ['blockchain', 'smart contracts'],
    'nft': ['blockchain', 'smart contracts'],
    
    # Game Development
    'unity': ['c#', 'game development'],
    'unreal engine': ['c++', 'game development'],
    'godot': ['gdscript', 'game development'],
    'game development': ['programming fundamentals', 'mathematics'],
    'opengl': ['graphics programming', 'c++'],
    'vulkan': ['opengl', 'advanced graphics'],
    
    # IoT & Embedded
    'arduino': ['c++', 'electronics'],
    'raspberry pi': ['linux', 'python'],
    'esp32': ['c++', 'microcontrollers'],
    'mqtt': ['networking', 'iot protocols'],
    'embedded programming': ['c', 'microcontrollers'],
    
    # Algorithms & Computer Science
    'algorithms': ['programming fundamentals', 'mathematics'],
    'data structures': ['programming fundamentals'],
    'big o notation': ['algorithms', 'mathematics'],
    'dynamic programming': ['algorithms', 'recursion'],
    'graph algorithms': ['data structures', 'algorithms'],
    
    # Software Engineering
    'design patterns': ['object oriented programming'],
    'clean code': ['programming fundamentals'],
    'tdd': ['testing concepts', 'programming'],
    'agile': ['software development'],
    'scrum': ['agile', 'project management'],
    'microservices': ['distributed systems', 'api design'],
    'domain driven design': ['software architecture', 'object oriented programming'],
    
    # Graphics & Multimedia
    'blender': ['3d modeling', 'animation'],
    'photoshop': ['image editing', 'digital art'],
    'after effects': ['video editing', 'motion graphics'],
    '3d modeling': ['mathematics', 'spatial reasoning'],
    'computer graphics': ['linear algebra', 'mathematics'],
    
    # Cybersecurity
    'penetration testing': ['networking', 'security fundamentals'],
    'cryptography': ['mathematics', 'security'],
    'ethical hacking': ['networking', 'security', 'linux'],
    'owasp': ['web security', 'application security'],
    'malware analysis': ['reverse engineering', 'security'],
}