TOPIC_CATEGORIES = {
    'programming_languages': [
        'python', 'javascript', 'typescript', 'java', 'c++', 'c', 'c#', 'go', 'rust',
        'kotlin', 'swift', 'ruby', 'php', 'scala', 'haskell', 'erlang', 'elixir',
        'clojure', 'f#', 'dart', 'lua', 'perl', 'r', 'matlab', 'julia', 'assembly',
        'cobol', 'fortran', 'pascal', 'ada', 'prolog', 'lisp', 'scheme', 'ocaml',
        'nim', 'zig', 'crystal', 'solidity', 'verilog', 'vhdl', 'v', 'odin', 'carbon',
        'mojo', 'gleam', 'roc', 'lean', 'chapel', 'racket', 'smalltalk', 'forth'
    ],
    
    'web_development': [
        'html', 'css', 'sass', 'less', 'stylus', 'postcss', 'tailwind css', 'bootstrap',
        'react', 'vue', 'angular', 'svelte', 'solid js', 'qwik', 'alpine js', 'lit',
        'nextjs', 'nuxt', 'gatsby', 'remix', 'astro', 'sveltekit', 'fresh', 'vite',
        'node.js', 'express', 'fastify', 'koa', 'hapi', 'nestjs', 'adonis js',
        'django', 'flask', 'fastapi', 'tornado', 'bottle', 'pyramid', 'chalice',
        'spring', 'spring boot', 'quarkus', 'micronaut', 'vert.x', 'dropwizard',
        'rails', 'sinatra', 'hanami', 'laravel', 'symfony', 'codeigniter', 'cakephp',
        'asp.net', 'blazor', 'mvc', 'web api', 'minimal apis', 'gin', 'echo', 'fiber',
        'actix', 'warp', 'rocket', 'axum', 'phoenix', 'plug', 'cowboy', 'strapi',
        'sanity', 'contentful', 'wordpress', 'drupal', 'joomla', 'ghost',
        'frontend', 'backend', 'fullstack', 'ssr', 'spa', 'pwa', 'jamstack',
        'web components', 'shadow dom', 'service workers', 'web assembly', 'wasm'
    ],
    
    'mobile_development': [
        'android', 'ios', 'flutter', 'react native', 'xamarin', 'ionic', 'cordova',
        'phonegap', 'unity mobile', 'kotlin multiplatform', 'swift ui', 'uikit',
        'jetpack compose', 'expo', 'capacitor', 'nativescript', 'titanium', 'maui',
        'corona sdk', 'cocos2d', 'defold', 'godot mobile', 'unreal mobile',
        'progressive web apps', 'tauri mobile', 'electron mobile', 'webview'
    ],
    
    'data_science_ml_ai': [
        'machine learning', 'deep learning', 'artificial intelligence', 'neural networks',
        'tensorflow', 'pytorch', 'keras', 'jax', 'flax', 'haiku', 'scikit-learn',
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'bokeh',
        'jupyter', 'jupyterlab', 'google colab', 'anaconda', 'miniconda', 'mamba',
        'mlflow', 'kubeflow', 'airflow', 'prefect', 'dagster', 'weights and biases',
        'neptune', 'clearml', 'comet', 'data analysis', 'statistics', 'regression',
        'classification', 'clustering', 'dimensionality reduction', 'feature engineering',
        'computer vision', 'nlp', 'natural language processing', 'reinforcement learning',
        'generative ai', 'llm', 'large language models', 'transformer', 'attention',
        'bert', 'gpt', 'claude', 'gemini', 'llama', 'mistral', 'stable diffusion',
        'midjourney', 'dall-e', 'gan', 'cnn', 'rnn', 'lstm', 'gru', 'attention mechanism',
        'ensemble methods', 'model deployment', 'mlops', 'automl', 'hyperparameter tuning',
        'cross validation', 'data preprocessing', 'data cleaning', 'data visualization',
        'time series analysis', 'forecasting', 'anomaly detection', 'recommendation systems',
        'a/b testing', 'causal inference', 'bayesian statistics', 'monte carlo methods',
        'gradient descent', 'backpropagation', 'optimization', 'regularization'
    ],
    
    'llm_ai_models_2025': [
        # OpenAI Models
        'gpt-4o', 'gpt-4o mini', 'gpt-4.1', 'gpt-4 turbo', 'gpt-4', 'gpt-3.5-turbo',
        'o1', 'o1-mini', 'o1-preview', 'o3', 'o3-mini', 'chatgpt', 'dall-e 3', 'sora',
        
        # Anthropic Models  
        'claude 4 opus', 'claude 4 sonnet', 'claude 3.7 sonnet', 'claude 3.5 sonnet',
        'claude 3.5 haiku', 'claude 3 opus', 'claude 3 sonnet', 'claude 3 haiku',
        'claude 2.1', 'claude 2.0', 'claude instant',
        
        # Google Models
        'gemini 2.5 pro', 'gemini 2.5 flash', 'gemini 2.0 flash', 'gemini 1.5 pro',
        'gemini 1.5 flash', 'gemini 1.0 ultra', 'gemini 1.0 pro', 'gemini 1.0 nano',
        'gemma 3 27b', 'gemma 3 12b', 'gemma 3 4b', 'gemma 3 1b', 'gemma 2 27b',
        'gemma 2 9b', 'gemma 2 2b', 'palm-2', 'bard',
        
        # Meta Models
        'llama 4', 'llama 4 scout', 'llama 4 maverick', 'llama 4 behemoth',
        'llama 3.3', 'llama 3.2', 'llama 3.1', 'llama 3', 'llama 2', 'code llama',
        
        # Mistral Models
        'mistral large 2', 'mistral small 3', 'mistral medium 3', 'mistral nemo',
        'mistral 7b', 'codestral', 'devstral', 'ministral 8b', 'ministral 3b',
        'mixtral 8x7b', 'mixtral 8x22b',
        
        # Other Major Models
        'deepseek v3', 'deepseek r1', 'deepseek coder', 'qwen 2.5', 'qwen qwq',
        'phi 4', 'phi 3.5', 'phi 3', 'cohere command r+', 'cohere command r',
        'cohere aya', 'nvidia nemotron', 'yi large', 'baichuan', 'chatglm',
        'intern lm', 'aquila', 'moss', 'spark', 'ernie', 'tongyi qianwen',
        
        # Specialized Models
        'whisper', 'tts', 'embeddings', 'text-embedding-ada-002', 'text-davinci-003',
        'gpt-3.5-turbo-instruct', 'text-moderation', 'janus pro', 'veo 2',
        
        # Model Context Protocol
        'mcp', 'model context protocol', 'mcp servers', 'mcp clients'
    ],
    
    'package_managers': [
        # JavaScript/Node.js
        'npm', 'yarn', 'yarn berry', 'yarn 2', 'yarn 3', 'yarn 4', 'pnpm', 'bun',
        'npx', 'bunx', 'volta', 'fnm', 'nvm', 'n', 'rush', 'lerna', 'nx',
        
        # Python  
        'pip', 'uv', 'poetry', 'pipenv', 'conda', 'mamba', 'micromamba', 'pdm',
        'hatch', 'setuptools', 'wheel', 'twine', 'flit', 'build', 'pipx',
        'virtualenv', 'venv', 'pyenv', 'anaconda', 'miniconda',
        
        # Rust
        'cargo', 'cargo-edit', 'cargo-update', 'cargo-audit', 'cargo-outdated',
        
        # Go
        'go mod', 'go get', 'dep', 'glide', 'godep', 'vendor',
        
        # Java/JVM
        'maven', 'gradle', 'sbt', 'ant', 'ivy', 'mill',
        
        # .NET
        'nuget', 'dotnet', 'paket',
        
        # PHP
        'composer', 'pear', 'pecl',
        
        # Ruby
        'gem', 'bundler', 'rbenv', 'rvm',
        
        # Other
        'brew', 'homebrew', 'apt', 'yum', 'dnf', 'pacman', 'zypper', 'portage',
        'snap', 'flatpak', 'appimage', 'chocolatey', 'scoop', 'winget',
        'conan', 'vcpkg', 'cpm', 'hunter'
    ],
    
    'databases': [
        # SQL Databases
        'mysql', 'postgresql', 'sqlite', 'oracle', 'sql server', 'mariadb',
        'db2', 'sybase', 'teradata', 'snowflake', 'bigquery', 'redshift',
        'cockroachdb', 'yugabytedb', 'planetscale', 'neon', 'supabase',
        'turso', 'libsql', 'duckdb', 'clickhouse', 'apache drill',
        
        # NoSQL Databases
        'mongodb', 'cassandra', 'couchdb', 'couchbase', 'rethinkdb',
        'arangodb', 'orientdb', 'neo4j', 'dgraph', 'amazon neptune',
        'tigergraph', 'janusgraph', 'apache giraph',
        
        # Key-Value Stores
        'redis', 'memcached', 'etcd', 'consul', 'zookeeper', 'riak',
        'amazon dynamodb', 'azure cosmos db', 'foundationdb',
        
        # Search Engines
        'elasticsearch', 'opensearch', 'solr', 'sphinx', 'whoosh',
        'meilisearch', 'typesense', 'algolia', 'swiftype',
        
        # Time Series
        'influxdb', 'prometheus', 'grafana', 'timescaledb', 'questdb',
        'kdb+', 'opentsdb', 'druid', 'pinot',
        
        # Firebase & BaaS
        'firebase', 'firestore', 'realtime database', 'appwrite', 'pocketbase',
        'hasura', 'directus', 'strapi', 'sanity', 'contentful',
        
        # Database Tools & Concepts
        'sql', 'nosql', 'acid', 'base', 'cap theorem', 'database design',
        'normalization', 'denormalization', 'indexing', 'query optimization',
        'sharding', 'replication', 'backup', 'migration', 'data warehouse',
        'etl', 'elt', 'olap', 'oltp', 'data lake', 'data mesh'
    ],
    
    'cloud_computing': [
        # Major Cloud Providers
        'aws', 'amazon web services', 'azure', 'microsoft azure', 'gcp', 'google cloud',
        'oracle cloud', 'ibm cloud', 'alibaba cloud', 'tencent cloud', 'huawei cloud',
        
        # Alternative Cloud Providers
        'digitalocean', 'linode', 'vultr', 'hetzner', 'scaleway', 'upcloud',
        'ovh', 'kimsufi', 'contabo', 'rackspace', 'brightbox',
        
        # Serverless & Platform Services
        'vercel', 'netlify', 'cloudflare', 'surge', 'github pages', 'gitlab pages',
        'heroku', 'railway', 'render', 'fly.io', 'cyclic', 'koyeb', 'deta',
        'supabase', 'appwrite', 'firebase', 'amplify',
        
        # AWS Services
        'lambda', 'ec2', 's3', 'rds', 'dynamodb', 'cloudfront', 'route53',
        'iam', 'vpc', 'cloudwatch', 'cloudformation', 'ecs', 'eks', 'fargate',
        'api gateway', 'cognito', 'sns', 'sqs', 'eventbridge', 'step functions',
        
        # Azure Services
        'azure functions', 'app service', 'blob storage', 'sql database',
        'cosmos db', 'active directory', 'key vault', 'application insights',
        
        # GCP Services
        'cloud functions', 'app engine', 'compute engine', 'cloud storage',
        'cloud sql', 'firestore', 'pub/sub', 'cloud run', 'gke',
        
        # Concepts
        'iaas', 'paas', 'saas', 'faas', 'baas', 'cloud architecture', 'cloud security',
        'cloud native', 'hybrid cloud', 'multi cloud', 'edge computing', 'cdn',
        'load balancing', 'auto scaling', 'cloud storage', 'cloud functions',
        'serverless', 'microservices', 'containers', 'orchestration'
    ],
    
    'devops_infrastructure': [
        # Containerization
        'docker', 'podman', 'containerd', 'cri-o', 'buildah', 'skopeo',
        'docker compose', 'docker swarm', 'buildkit', 'kaniko', 'img',
        
        # Orchestration
        'kubernetes', 'k8s', 'openshift', 'rancher', 'nomad', 'docker swarm',
        'helm', 'kustomize', 'operator', 'istio', 'linkerd', 'consul connect',
        
        # CI/CD
        'jenkins', 'gitlab ci', 'github actions', 'circleci', 'travis ci',
        'buildkite', 'drone', 'tekton', 'argo cd', 'flux', 'spinnaker',
        'azure devops', 'bamboo', 'teamcity', 'codebuild', 'codepipeline',
        
        # Infrastructure as Code
        'terraform', 'terragrunt', 'pulumi', 'cloudformation', 'cdk',
        'bicep', 'ansible', 'puppet', 'chef', 'saltstack', 'vagrant',
        'packer', 'crossplane', 'cdktf',
        
        # Service Discovery & Configuration
        'consul', 'etcd', 'zookeeper', 'vault', 'secrets manager',
        'parameter store', 'config maps', 'helm charts',
        
        # Monitoring & Observability
        'prometheus', 'grafana', 'jaeger', 'zipkin', 'opentelemetry',
        'elk stack', 'elasticsearch', 'logstash', 'kibana', 'fluentd',
        'fluent bit', 'loki', 'tempo', 'cortex', 'thanos', 'vector',
        'datadog', 'new relic', 'splunk', 'dynatrace', 'appdynamics',
        'pingdom', 'uptimerobot', 'statuspage', 'pagerduty', 'opsgenie',
        
        # Security & Scanning
        'trivy', 'clair', 'snyk', 'aqua', 'twistlock', 'falco', 'opa',
        'gatekeeper', 'policy as code', 'rbac', 'network policies',
        
        # GitOps & Deployment
        'gitops', 'argocd', 'flux', 'tekton', 'jenkins x', 'skaffold',
        'draft', 'brigade', 'captain', 'werf', 'okteto', 'garden',
        
        # Concepts
        'ci/cd', 'continuous integration', 'continuous deployment', 'continuous delivery',
        'infrastructure as code', 'configuration management', 'blue-green deployment',
        'canary deployment', 'rolling deployment', 'feature flags', 'chaos engineering',
        'site reliability engineering', 'sre', 'devops', 'platform engineering',
        'observability', 'monitoring', 'logging', 'tracing', 'alerting'
    ],
    
    'cybersecurity': [
        # General Security
        'cybersecurity', 'information security', 'infosec', 'application security',
        'network security', 'web security', 'mobile security', 'cloud security',
        'container security', 'api security', 'iot security', 'ot security',
        
        # Cryptography
        'cryptography', 'encryption', 'decryption', 'hashing', 'digital signatures',
        'pki', 'certificates', 'ssl', 'tls', 'https', 'aes', 'rsa', 'ecc',
        'diffie-hellman', 'sha', 'md5', 'bcrypt', 'scrypt', 'argon2',
        'zero knowledge proofs', 'homomorphic encryption', 'quantum cryptography',
        
        # Authentication & Authorization
        'authentication', 'authorization', 'oauth', 'oauth2', 'openid connect',
        'saml', 'jwt', 'session management', 'multi-factor authentication', 'mfa',
        '2fa', 'biometrics', 'passwordless', 'webauthn', 'fido2', 'passkeys',
        'single sign-on', 'sso', 'identity management', 'access control', 'rbac',
        'abac', 'zero trust', 'principle of least privilege',
        
        # Vulnerability Assessment & Testing
        'penetration testing', 'pentest', 'vulnerability assessment', 'bug bounty',
        'red team', 'blue team', 'purple team', 'ethical hacking', 'white hat',
        'black hat', 'grey hat', 'social engineering', 'phishing', 'whaling',
        'spear phishing', 'pretexting', 'baiting', 'quid pro quo',
        
        # Security Tools
        'nmap', 'wireshark', 'metasploit', 'burp suite', 'owasp zap', 'sqlmap',
        'nikto', 'dirb', 'gobuster', 'ffuf', 'hydra', 'john the ripper',
        'hashcat', 'aircrack-ng', 'nessus', 'openvas', 'qualys', 'rapid7',
        'snort', 'suricata', 'ossec', 'wazuh', 'osquery', 'yara',
        
        # Threats & Attacks
        'malware', 'virus', 'trojan', 'ransomware', 'spyware', 'adware',
        'rootkit', 'botnet', 'ddos', 'dos', 'sql injection', 'xss',
        'cross-site scripting', 'csrf', 'cross-site request forgery', 'lfi',
        'rfi', 'directory traversal', 'command injection', 'xxe',
        'deserialization', 'buffer overflow', 'race condition', 'privilege escalation',
        'lateral movement', 'persistence', 'exfiltration', 'c2', 'apt',
        
        # Security Frameworks & Standards
        'owasp', 'owasp top 10', 'nist', 'iso 27001', 'soc 2', 'pci dss',
        'gdpr', 'hipaa', 'sox', 'fisma', 'common criteria', 'cvss', 'cve',
        'cwe', 'sans top 25', 'mitre att&ck', 'cyber kill chain',
        'diamond model', 'pyramid of pain',
        
        # Security Operations
        'soc', 'security operations center', 'siem', 'soar', 'xdr', 'edr',
        'incident response', 'digital forensics', 'threat hunting', 'threat intelligence',
        'ioc', 'indicators of compromise', 'ttps', 'tactics techniques procedures',
        'sandbox', 'honeypot', 'deception technology', 'threat modeling',
        
        # Network Security
        'firewall', 'ids', 'ips', 'intrusion detection', 'intrusion prevention',
        'waf', 'web application firewall', 'ngfw', 'next generation firewall',
        'utm', 'unified threat management', 'vpn', 'virtual private network',
        'ipsec', 'wireguard', 'openvpn', 'network segmentation', 'micro-segmentation',
        'zero trust network', 'sdn security', 'network access control', 'nac'
    ],
    
    'networking': [
        # Protocols
        'tcp/ip', 'http', 'https', 'http/2', 'http/3', 'websockets', 'grpc',
        'dns', 'dhcp', 'nat', 'pat', 'icmp', 'arp', 'ospf', 'bgp', 'rip',
        'eigrp', 'mpls', 'vxlan', 'gre', 'ipsec', 'l2tp', 'pptp', 'ssh',
        'telnet', 'ftp', 'sftp', 'smtp', 'pop3', 'imap', 'snmp', 'ntp',
        'radius', 'tacacs+', 'ldap', 'kerberos', 'oauth', 'saml',
        
        # Network Infrastructure
        'router', 'switch', 'hub', 'bridge', 'gateway', 'firewall', 'load balancer',
        'proxy', 'reverse proxy', 'cdn', 'content delivery network', 'wan',
        'lan', 'man', 'pan', 'vpn', 'vlan', 'vrf', 'stp', 'rstp', 'mstp',
        'lacp', 'etherchannel', 'port channel', 'link aggregation',
        
        # Network Architecture
        'osi model', 'tcp/ip model', 'network topology', 'star topology',
        'mesh topology', 'ring topology', 'bus topology', 'hybrid topology',
        'subnetting', 'vlsm', 'cidr', 'supernetting', 'route summarization',
        'network design', 'hierarchical design', 'three-tier architecture',
        'spine-leaf architecture', 'clos network', 'fat tree',
        
        # Wireless Networking
        'wifi', 'wireless', '802.11', 'wpa', 'wpa2', 'wpa3', 'wep', 'bluetooth',
        'zigbee', 'z-wave', 'lora', 'lorawan', 'cellular', '3g', '4g', '5g',
        'lte', 'gsm', 'cdma', 'mimo', 'beamforming', 'mesh networking',
        
        # Network Monitoring & Troubleshooting
        'wireshark', 'tcpdump', 'netstat', 'ss', 'ping', 'traceroute', 'mtr',
        'nmap', 'nslookup', 'dig', 'iperf', 'bandwidth', 'latency', 'jitter',
        'packet loss', 'network performance', 'qos', 'traffic shaping',
        'network forensics', 'flow analysis', 'netflow', 'sflow', 'ipfix',
        
        # Modern Networking
        'sdn', 'software defined networking', 'nfv', 'network function virtualization',
        'overlay networks', 'underlay networks', 'network virtualization',
        'cloud networking', 'hybrid cloud networking', 'multi-cloud networking',
        'edge computing', 'edge networking', 'intent-based networking', 'ibn',
        'network automation', 'network orchestration', 'network as code'
    ],
    
    'operating_systems': [
        # Linux Distributions
        'linux', 'ubuntu', 'debian', 'centos', 'rhel', 'red hat', 'fedora',
        'suse', 'opensuse', 'arch linux', 'manjaro', 'gentoo', 'slackware',
        'mint', 'elementary os', 'pop os', 'zorin os', 'kali linux',
        'parrot os', 'blackarch', 'tails', 'alpine linux', 'void linux',
        'nixos', 'guix', 'clear linux', 'coreos', 'flatcar', 'rancher os',
        'k3os', 'bottlerocket', 'photon os', 'amazon linux',
        
        # Unix Systems
        'unix', 'aix', 'solaris', 'hp-ux', 'irix', 'freebsd', 'openbsd',
        'netbsd', 'dragonflybsd', 'trueos', 'ghostbsd', 'hardenedbsd',
        
        # Windows
        'windows', 'windows 11', 'windows 10', 'windows server', 'windows subsystem for linux',
        'wsl', 'wsl2', 'powershell', 'cmd', 'batch', 'windows terminal',
        'hyper-v', 'iis', 'active directory', 'group policy',
        
        # macOS
        'macos', 'mac os x', 'darwin', 'homebrew', 'macports', 'xcode',
        'terminal', 'zsh', 'bash', 'fish', 'iterm2', 'finder',
        
        # Mobile OS
        'android', 'ios', 'ipados', 'watchos', 'tvos', 'wear os', 'tizen',
        'sailfish os', 'ubuntu touch', 'postmarket os', 'lineage os',
        'graphene os', 'calyx os', 'e foundation', 'pure os',
        
        # Embedded & Real-time OS
        'freertos', 'rtos', 'vxworks', 'qnx', 'integrity', 'threadx',
        'zephyr', 'mbed os', 'riot', 'contiki', 'tinyos', 'nucleus',
        'micrium', 'embos', 'rtems', 'xenomai', 'rt-linux',
        
        # System Components
        'kernel', 'bootloader', 'grub', 'systemd', 'init', 'upstart',
        'openrc', 'runit', 'shell', 'bash', 'zsh', 'fish', 'csh', 'tcsh',
        'ksh', 'dash', 'command line', 'terminal', 'console', 'tty',
        'process management', 'memory management', 'file systems',
        'ext4', 'xfs', 'btrfs', 'zfs', 'ntfs', 'fat32', 'exfat', 'apfs',
        'permissions', 'users', 'groups', 'sudo', 'su', 'chmod', 'chown',
        'cron', 'crontab', 'systemd timers', 'at', 'batch',
        
        # Virtualization
        'virtualization', 'hypervisor', 'type 1 hypervisor', 'type 2 hypervisor',
        'vmware', 'virtualbox', 'kvm', 'qemu', 'xen', 'hyper-v', 'parallels',
        'proxmox', 'esxi', 'vcenter', 'libvirt', 'vagrant', 'packer'
    ],
    
    'algorithms_data_structures': [
        # Data Structures
        'array', 'linked list', 'doubly linked list', 'circular linked list',
        'stack', 'queue', 'deque', 'priority queue', 'heap', 'binary heap',
        'fibonacci heap', 'binomial heap', 'tree', 'binary tree', 'bst',
        'binary search tree', 'avl tree', 'red-black tree', 'b-tree',
        'b+ tree', 'trie', 'suffix tree', 'segment tree', 'fenwick tree',
        'disjoint set', 'union find', 'graph', 'directed graph', 'undirected graph',
        'weighted graph', 'adjacency matrix', 'adjacency list', 'hash table',
        'hash map', 'hash set', 'bloom filter', 'skip list',
        
        # Sorting Algorithms
        'bubble sort', 'selection sort', 'insertion sort', 'merge sort',
        'quick sort', 'heap sort', 'radix sort', 'counting sort', 'bucket sort',
        'shell sort', 'cocktail sort', 'gnome sort', 'tim sort', 'intro sort',
        
        # Searching Algorithms
        'linear search', 'binary search', 'interpolation search', 'exponential search',
        'jump search', 'fibonacci search', 'ternary search', 'depth first search',
        'dfs', 'breadth first search', 'bfs', 'a* search', 'dijkstra', 'bellman ford',
        'floyd warshall', 'kruskal', 'prim', 'topological sort',
        
        # Graph Algorithms
        'graph traversal', 'shortest path', 'minimum spanning tree', 'mst',
        'strongly connected components', 'articulation points', 'bridges',
        'maximum flow', 'ford fulkerson', 'edmonds karp', 'bipartite matching',
        'network flow', 'traveling salesman', 'tsp', 'hamiltonian path',
        'eulerian path', 'cycle detection', 'graph coloring',
        
        # Dynamic Programming
        'dynamic programming', 'memoization', 'tabulation', 'optimal substructure',
        'overlapping subproblems', 'knapsack problem', 'longest common subsequence',
        'longest increasing subsequence', 'edit distance', 'coin change',
        'fibonacci', 'factorial', 'catalan numbers', 'matrix chain multiplication',
        
        # String Algorithms
        'string matching', 'kmp algorithm', 'rabin karp', 'boyer moore',
        'z algorithm', 'suffix array', 'lcp array', 'manacher algorithm',
        'regular expressions', 'finite automata', 'context free grammar',
        
        # Recursion & Divide and Conquer
        'recursion', 'tail recursion', 'divide and conquer', 'master theorem',
        'backtracking', 'branch and bound', 'n queens', 'sudoku solver',
        'maze solving', 'permutations', 'combinations', 'subset generation',
        
        # Greedy Algorithms
        'greedy algorithms', 'activity selection', 'fractional knapsack',
        'huffman coding', 'job scheduling', 'minimum coins', 'gas station',
        
        # Computational Complexity
        'big o notation', 'time complexity', 'space complexity', 'asymptotic analysis',
        'worst case', 'best case', 'average case', 'amortized analysis',
        'p vs np', 'np complete', 'np hard', 'polynomial time', 'exponential time',
        'logarithmic time', 'linear time', 'quadratic time', 'cubic time',
        
        # Advanced Algorithms
        'approximation algorithms', 'randomized algorithms', 'monte carlo',
        'las vegas algorithms', 'parallel algorithms', 'distributed algorithms',
        'online algorithms', 'streaming algorithms', 'external sorting',
        'cache oblivious algorithms', 'succinct data structures'
    ],
    
    'software_engineering': [
        # Development Methodologies
        'software development', 'sdlc', 'software development life cycle',
        'agile', 'scrum', 'kanban', 'lean', 'waterfall', 'spiral', 'v-model',
        'rad', 'rapid application development', 'extreme programming', 'xp',
        'feature driven development', 'fdd', 'crystal', 'dsdm',
        'scaled agile', 'safe', 'less', 'nexus', 'spotify model',
        
        # Design Patterns
        'design patterns', 'gang of four', 'gof', 'creational patterns',
        'singleton', 'factory', 'abstract factory', 'builder', 'prototype',
        'structural patterns', 'adapter', 'bridge', 'composite', 'decorator',
        'facade', 'flyweight', 'proxy', 'behavioral patterns', 'observer',
        'strategy', 'command', 'state', 'template method', 'visitor',
        'mediator', 'memento', 'chain of responsibility', 'iterator',
        'mvc', 'mvp', 'mvvm', 'repository pattern', 'dependency injection',
        
        # Software Architecture
        'software architecture', 'architectural patterns', 'layered architecture',
        'microservices', 'monolith', 'service oriented architecture', 'soa',
        'event driven architecture', 'cqrs', 'event sourcing', 'hexagonal architecture',
        'clean architecture', 'onion architecture', 'ports and adapters',
        'domain driven design', 'ddd', 'bounded context', 'aggregate',
        'saga pattern', 'circuit breaker', 'bulkhead', 'strangler fig',
        
        # Software Quality & Testing
        'software testing', 'unit testing', 'integration testing', 'system testing',
        'acceptance testing', 'regression testing', 'performance testing',
        'load testing', 'stress testing', 'security testing', 'usability testing',
        'smoke testing', 'sanity testing', 'exploratory testing', 'manual testing',
        'automated testing', 'test driven development', 'tdd', 'behavior driven development',
        'bdd', 'acceptance test driven development', 'atdd', 'test pyramid',
        'test doubles', 'mocks', 'stubs', 'spies', 'fakes', 'dummies',
        'code coverage', 'mutation testing', 'property based testing',
        
        # Code Quality
        'clean code', 'code smells', 'refactoring', 'technical debt',
        'code review', 'pair programming', 'mob programming', 'static analysis',
        'linting', 'code formatting', 'naming conventions', 'documentation',
        'comments', 'self documenting code', 'readable code', 'maintainable code',
        
        # Software Principles
        'solid principles', 'single responsibility', 'open closed', 'liskov substitution',
        'interface segregation', 'dependency inversion', 'dry', 'dont repeat yourself',
        'kiss', 'keep it simple stupid', 'yagni', 'you arent gonna need it',
        'separation of concerns', 'loose coupling', 'high cohesion',
        'composition over inheritance', 'favor composition', 'tell dont ask',
        'law of demeter', 'principle of least knowledge',
        
        # Project Management
        'project management', 'sprint planning', 'backlog', 'user stories',
        'epic', 'feature', 'story points', 'velocity', 'burndown chart',
        'retrospective', 'daily standup', 'sprint review', 'definition of done',
        'acceptance criteria', 'product owner', 'scrum master', 'stakeholders',
        
        # Documentation
        'technical documentation', 'api documentation', 'user documentation',
        'code documentation', 'architecture documentation', 'design documents',
        'requirements specification', 'functional specification', 'technical specification',
        'readme', 'changelog', 'release notes', 'troubleshooting guide',
        'runbook', 'playbook', 'wiki', 'knowledge base'
    ],
    
    'game_development': [
        # Game Engines
        'unity', 'unreal engine', 'godot', 'game maker studio', 'construct',
        'rpg maker', 'renpy', 'twine', 'bitsy', 'pico-8', 'love2d', 'defold',
        'cocos2d', 'cocos creator', 'amazon lumberyard', 'cryengine',
        'source engine', 'id tech', 'frostbite', 'anvil', 'creation engine',
        'bevy', 'amethyst', 'panda3d', 'irrlicht', 'ogre3d', 'haxeflixel',
        'phaser', 'pixi.js', 'three.js', 'babylon.js', 'playcanvas',
        
        # Graphics Programming
        'graphics programming', 'rendering', 'shaders', 'vertex shader',
        'fragment shader', 'pixel shader', 'compute shader', 'hlsl', 'glsl',
        'spirv', 'opengl', 'directx', 'vulkan', 'metal', 'webgl', 'webgpu',
        'ray tracing', 'rasterization', 'texture mapping', 'normal mapping',
        'bump mapping', 'displacement mapping', 'pbr', 'physically based rendering',
        'lighting', 'shadows', 'global illumination', 'ambient occlusion',
        'screen space reflections', 'temporal anti-aliasing', 'fxaa', 'msaa',
        
        # Game Programming
        'game loop', 'update loop', 'render loop', 'delta time', 'frame rate',
        'fps', 'vsync', 'input handling', 'event system', 'state management',
        'scene management', 'entity component system', 'ecs', 'game objects',
        'components', 'systems', 'scripting', 'lua scripting', 'visual scripting',
        'blueprint', 'gd script', 'c# scripting', 'javascript scripting',
        
        # Game Physics
        'game physics', 'collision detection', 'collision response', 'rigid body',
        'soft body', 'particle systems', 'fluid simulation', 'cloth simulation',
        'ragdoll physics', 'inverse kinematics', 'forward kinematics',
        'physics engines', 'box2d', 'bullet physics', 'havok', 'physx',
        'ode', 'newton dynamics', 'chipmunk', 'matter.js', 'cannon.js',
        
        # Game AI
        'game ai', 'pathfinding', 'a* pathfinding', 'navigation mesh', 'navmesh',
        'behavior trees', 'finite state machines', 'decision trees', 'fuzzy logic',
        'neural networks in games', 'genetic algorithms', 'flocking', 'steering behaviors',
        'crowd simulation', 'procedural generation', 'procedural content generation',
        'noise functions', 'perlin noise', 'simplex noise', 'cellular automata',
        
        # Game Audio
        'game audio', 'sound effects', 'music', 'ambient sound', '3d audio',
        'spatial audio', 'audio occlusion', 'audio reverb', 'dynamic music',
        'adaptive audio', 'wwise', 'fmod', 'audio middleware', 'compression',
        'audio streaming', 'audio synthesis', 'procedural audio',
        
        # Game Design
        'game design', 'level design', 'gameplay mechanics', 'game balance',
        'difficulty curve', 'player progression', 'reward systems', 'achievements',
        'user interface', 'user experience', 'accessibility', 'game feel',
        'juice', 'polish', 'playtesting', 'user research', 'analytics',
        'monetization', 'free to play', 'in app purchases', 'gacha', 'loot boxes',
        
        # Multiplayer & Networking
        'multiplayer', 'networking', 'client server', 'peer to peer', 'p2p',
        'authoritative server', 'lag compensation', 'prediction', 'rollback',
        'interpolation', 'extrapolation', 'delta compression', 'reliable udp',
        'photon', 'mirror networking', 'unity netcode', 'dedicated servers',
        'matchmaking', 'lobbies', 'real time multiplayer', 'turn based multiplayer',
        
        # Mobile Game Development
        'mobile games', 'android games', 'ios games', 'touch controls',
        'accelerometer', 'gyroscope', 'ar games', 'augmented reality',
        'arkit', 'arcore', 'vuforia', 'performance optimization',
        'battery optimization', 'thermal throttling', 'in app billing',
        
        # VR/AR Development
        'virtual reality', 'vr', 'augmented reality', 'ar', 'mixed reality', 'mr',
        'oculus', 'meta quest', 'htc vive', 'valve index', 'pico', 'hololens',
        'magic leap', 'steamvr', 'openvr', 'openxr', 'hand tracking',
        'eye tracking', 'room scale', 'teleportation', 'locomotion',
        'vr ui', 'spatial ui', 'haptic feedback', 'motion sickness',
        
        # Game Platforms
        'steam', 'epic games store', 'gog', 'itch.io', 'game jolt', 'xbox',
        'playstation', 'nintendo switch', 'mobile platforms', 'web games',
        'browser games', 'console development', 'pc gaming', 'cloud gaming',
        'stadia', 'luna', 'geforce now', 'xbox cloud gaming'
    ],
    
    'computer_graphics': [
        # 3D Graphics
        '3d graphics', '3d modeling', '3d animation', 'polygonal modeling',
        'nurbs', 'subdivision surfaces', 'sculpting', 'retopology',
        'uv mapping', 'texturing', 'materials', 'shaders', 'lighting',
        'rendering', 'ray tracing', 'path tracing', 'global illumination',
        'radiosity', 'photon mapping', 'ambient occlusion', 'subsurface scattering',
        
        # 2D Graphics
        '2d graphics', 'raster graphics', 'vector graphics', 'pixel art',
        'digital painting', 'illustration', 'concept art', 'character design',
        'environment design', 'ui design', 'icon design', 'typography',
        'color theory', 'composition', 'perspective', 'anatomy',
        
        # Software & Tools
        'blender', 'maya', '3ds max', 'cinema 4d', 'houdini', 'zbrush',
        'substance painter', 'substance designer', 'quixel mixer', 'marvelous designer',
        'photoshop', 'illustrator', 'after effects', 'premiere pro', 'davinci resolve',
        'nuke', 'fusion', 'houdini', 'katana', 'arnold', 'v-ray', 'cycles',
        'eevee', 'octane', 'redshift', 'corona', 'keyshot', 'marmoset toolbag',
        
        # Animation
        'animation', 'keyframe animation', 'motion capture', 'mocap',
        'procedural animation', 'rigging', 'skinning', 'weight painting',
        'inverse kinematics', 'forward kinematics', 'constraints', 'drivers',
        'bone systems', 'armatures', 'facial animation', 'lip sync',
        'particle animation', 'cloth simulation', 'fluid simulation',
        'destruction simulation', 'crowd simulation', 'hair simulation',
        
        # Rendering Techniques
        'rendering pipeline', 'vertex processing', 'primitive assembly',
        'rasterization', 'fragment processing', 'depth testing', 'alpha blending',
        'anti-aliasing', 'msaa', 'fxaa', 'smaa', 'temporal anti-aliasing',
        'deferred rendering', 'forward rendering', 'tiled rendering',
        'clustered rendering', 'visibility culling', 'frustum culling',
        'occlusion culling', 'level of detail', 'lod', 'imposters',
        
        # Computer Vision
        'computer vision', 'image processing', 'feature detection',
        'edge detection', 'corner detection', 'blob detection',
        'template matching', 'optical flow', 'structure from motion',
        'stereo vision', 'depth estimation', 'camera calibration',
        'image segmentation', 'object tracking', 'face detection',
        'facial recognition', 'gesture recognition', 'pose estimation',
        
        # Graphics APIs
        'opengl', 'directx', 'vulkan', 'metal', 'webgl', 'webgpu',
        'opencl', 'cuda', 'compute shaders', 'graphics drivers',
        'gpu programming', 'parallel computing', 'simd', 'vector processing',
        
        # Mathematical Foundations
        'linear algebra', 'matrix transformations', 'vectors', 'quaternions',
        'geometric transformations', 'projection matrices', 'view matrices',
        'model matrices', 'homogeneous coordinates', 'barycentric coordinates',
        'bezier curves', 'b-splines', 'catmull-rom splines', 'interpolation',
        'noise functions', 'perlin noise', 'simplex noise', 'fractal noise'
    ],
    
    'blockchain_crypto': [
        # Blockchain Fundamentals
        'blockchain', 'distributed ledger', 'decentralization', 'consensus',
        'proof of work', 'proof of stake', 'proof of authority', 'proof of history',
        'delegated proof of stake', 'mining', 'validators', 'nodes', 'full nodes',
        'light nodes', 'merkle tree', 'hash function', 'cryptographic hash',
        'digital signatures', 'public key cryptography', 'private key', 'wallet',
        
        # Cryptocurrencies
        'cryptocurrency', 'bitcoin', 'ethereum', 'binance coin', 'cardano',
        'solana', 'polkadot', 'avalanche', 'polygon', 'chainlink', 'litecoin',
        'bitcoin cash', 'ripple', 'xrp', 'stellar', 'monero', 'zcash', 'dash',
        'dogecoin', 'shiba inu', 'pepe', 'meme coins', 'stablecoins', 'usdc',
        'usdt', 'dai', 'busd', 'frax', 'terra luna', 'ust', 'algorithmic stablecoins',
        
        # Smart Contracts & DApps
        'smart contracts', 'solidity', 'vyper', 'rust', 'move', 'cairo',
        'web3', 'dapp', 'decentralized applications', 'ethereum virtual machine',
        'evm', 'gas', 'gas fees', 'gwei', 'transaction fees', 'contract deployment',
        'contract verification', 'proxy contracts', 'upgradeable contracts',
        'multisig', 'timelock', 'governance', 'dao', 'decentralized autonomous organization',
        
        # DeFi (Decentralized Finance)
        'defi', 'decentralized finance', 'automated market maker', 'amm',
        'liquidity pool', 'yield farming', 'liquidity mining', 'staking',
        'lending', 'borrowing', 'flash loans', 'arbitrage', 'dex',
        'decentralized exchange', 'uniswap', 'sushiswap', 'pancakeswap',
        'curve', 'balancer', 'compound', 'aave', 'makerdao', 'synthetix',
        'yearn finance', 'convex', 'frax finance', 'olympus dao',
        
        # NFTs & Digital Assets
        'nft', 'non-fungible token', 'erc-721', 'erc-1155', 'metadata',
        'ipfs', 'opensea', 'rarible', 'foundation', 'superrare', 'async art',
        'cryptopunks', 'bored ape yacht club', 'art blocks', 'generative art',
        'pfp', 'profile picture', 'utility nfts', 'gaming nfts', 'music nfts',
        'fractional ownership', 'nft marketplaces', 'royalties', 'creator economy',
        
        # Layer 2 & Scaling
        'layer 2', 'scaling solutions', 'rollups', 'optimistic rollups',
        'zk rollups', 'zero knowledge', 'zk-snarks', 'zk-starks', 'plasma',
        'state channels', 'lightning network', 'polygon', 'arbitrum', 'optimism',
        'starknet', 'zksync', 'loopring', 'immutable x', 'xdai', 'sidechains',
        
        # Blockchain Development
        'web3 development', 'truffle', 'hardhat', 'foundry', 'remix', 'ganache',
        'metamask', 'walletconnect', 'ethers.js', 'web3.js', 'wagmi', 'rainbow kit',
        'the graph', 'subgraph', 'indexing', 'oracles', 'chainlink', 'band protocol',
        'api3', 'tellor', 'uma', 'flux', 'nest protocol', 'pyth network',
        
        # Enterprise & Permissioned Blockchains
        'hyperledger fabric', 'hyperledger besu', 'r3 corda', 'jpmorgan quorum',
        'microsoft azure blockchain', 'aws blockchain', 'google cloud blockchain',
        'enterprise blockchain', 'permissioned blockchain', 'consortium blockchain',
        'supply chain', 'traceability', 'digital identity', 'self sovereign identity',
        
        # Security & Auditing
        'smart contract security', 'audit', 'formal verification', 'mythril',
        'slither', 'echidna', 'manticore', 'securify', 'oyente', 'certik',
        'openzeppelin', 'security patterns', 'reentrancy', 'flash loan attacks',
        'front running', 'mev', 'maximum extractable value', 'sandwich attacks',
        'rugs', 'rug pulls', 'exit scams', 'protocol risks', 'bridge hacks'
    ],
    
    'iot_embedded': [
        # IoT Platforms & Protocols
        'iot', 'internet of things', 'iiot', 'industrial iot', 'edge computing',
        'fog computing', 'mqtt', 'coap', 'http', 'websockets', 'lorawan',
        'sigfox', 'nbiot', 'lte-m', 'cellular iot', 'wifi', 'bluetooth',
        'bluetooth le', 'zigbee', 'z-wave', 'thread', 'matter', 'homekit',
        'alexa', 'google assistant', 'aws iot', 'azure iot', 'google cloud iot',
        
        # Microcontrollers & SoCs
        'microcontroller', 'microprocessor', 'mcu', 'soc', 'system on chip',
        'arduino', 'raspberry pi', 'esp32', 'esp8266', 'stm32', 'pic',
        'atmel', 'arm cortex', 'risc-v', 'avr', 'msp430', 'nordic nrf',
        'cypress psoc', 'ti cc', 'broadcom', 'qualcomm', 'mediatek',
        'rockchip', 'allwinner', 'amlogic', 'nvidia jetson', 'intel edison',
        
        # Sensors & Actuators
        'sensors', 'actuators', 'temperature sensor', 'humidity sensor',
        'pressure sensor', 'accelerometer', 'gyroscope', 'magnetometer',
        'imu', 'gps', 'camera', 'microphone', 'speaker', 'led', 'display',
        'lcd', 'oled', 'e-ink', 'servo motor', 'stepper motor', 'dc motor',
        'relay', 'solenoid', 'piezo', 'ultrasonic sensor', 'lidar', 'radar',
        'proximity sensor', 'light sensor', 'gas sensor', 'ph sensor',
        
        # Embedded Programming
        'embedded programming', 'c programming', 'c++', 'assembly language',
        'firmware', 'bootloader', 'rtos', 'real time operating system',
        'freertos', 'zephyr', 'riot', 'contiki', 'mbed os', 'arduino ide',
        'platformio', 'embedded linux', 'buildroot', 'yocto', 'openembedded',
        'cross compilation', 'toolchain', 'debugger', 'jtag', 'swd',
        
        # Communication & Networking
        'serial communication', 'uart', 'spi', 'i2c', 'can bus', 'rs485',
        'rs232', 'usb', 'ethernet', 'tcp/ip', 'udp', 'modbus', 'profibus',
        'profinet', 'ethercat', 'can open', 'bacnet', 'opc ua', 'industrial protocols',
        
        # Power Management
        'power management', 'low power', 'sleep modes', 'deep sleep',
        'power consumption', 'battery life', 'energy harvesting',
        'solar power', 'wireless charging', 'power supply', 'voltage regulation',
        'buck converter', 'boost converter', 'ldo', 'switching regulator',
        
        # Security
        'iot security', 'embedded security', 'secure boot', 'encryption',
        'authentication', 'pki', 'certificates', 'tls', 'dtls', 'secure element',
        'hardware security module', 'hsm', 'tpm', 'arm trustzone',
        'secure firmware updates', 'ota updates', 'code signing',
        
        # Industrial Applications
        'automation', 'industrial automation', 'scada', 'plc', 'hmi',
        'distributed control system', 'dcs', 'manufacturing execution system',
        'mes', 'predictive maintenance', 'condition monitoring', 'asset tracking',
        'inventory management', 'supply chain', 'smart factory', 'industry 4.0',
        
        # Smart Home & Consumer IoT
        'smart home', 'home automation', 'smart thermostat', 'smart lighting',
        'smart lock', 'security camera', 'smart speaker', 'smart tv',
        'wearables', 'fitness tracker', 'smartwatch', 'health monitoring',
        'elderly care', 'pet tracking', 'smart agriculture', 'precision farming'
    ],
    
    'browsers_web_tools': [
        # Web Browsers
        'chrome', 'firefox', 'safari', 'edge', 'opera', 'brave', 'vivaldi',
        'chromium', 'webkit', 'gecko', 'blink', 'quantum', 'lynx', 'links',
        'elinks', 'w3m', 'text browsers', 'headless browsers', 'puppeteer',
        'playwright', 'selenium', 'cypress', 'webdriver', 'browser automation',
        
        # Browser Extensions
        'browser extensions', 'chrome extensions', 'firefox addons', 'safari extensions',
        'webextensions api', 'manifest v3', 'content scripts', 'background scripts',
        'popup', 'options page', 'permissions', 'cross-origin requests',
        
        # Web Development Tools
        'developer tools', 'devtools', 'inspector', 'console', 'debugger',
        'network tab', 'performance tab', 'memory tab', 'security tab',
        'lighthouse', 'pagespeed insights', 'web vitals', 'accessibility audit',
        'seo audit', 'performance audit', 'best practices audit',
        
        # Build Tools & Bundlers
        'webpack', 'vite', 'rollup', 'parcel', 'esbuild', 'swc', 'babel',
        'typescript compiler', 'tsc', 'sass', 'less', 'postcss', 'autoprefixer',
        'cssnano', 'uglify', 'terser', 'minification', 'tree shaking',
        'code splitting', 'hot module replacement', 'hmr', 'live reload',
        
        # Testing Tools
        'jest', 'vitest', 'mocha', 'jasmine', 'karma', 'cypress', 'playwright',
        'puppeteer', 'selenium', 'webdriverio', 'testcafe', 'nightwatch',
        'protractor', 'storybook', 'chromatic', 'percy', 'applitools',
        'visual regression testing', 'snapshot testing', 'e2e testing',
        
        # Linting & Formatting
        'eslint', 'tslint', 'jshint', 'jslint', 'prettier', 'stylelint',
        'htmlhint', 'markdownlint', 'editorconfig', 'husky', 'lint-staged',
        'pre-commit hooks', 'commitlint', 'conventional commits',
        
        # Package Management & Registries
        'npm registry', 'yarn registry', 'github packages', 'private registries',
        'verdaccio', 'nexus', 'artifactory', 'package-lock.json', 'yarn.lock',
        'pnpm-lock.yaml', 'shrinkwrap', 'lock files', 'semantic versioning',
        'semver', 'npm audit', 'vulnerability scanning', 'license checking',
        
        # Web Standards & APIs
        'web standards', 'w3c', 'whatwg', 'ecmascript', 'html5', 'css3',
        'web apis', 'fetch api', 'websockets api', 'service worker api',
        'web workers', 'shared workers', 'broadcast channel', 'indexeddb',
        'localstorage', 'sessionstorage', 'cache api', 'push api',
        'notifications api', 'geolocation api', 'camera api', 'microphone api',
        'webrtc', 'web assembly', 'wasm', 'web components', 'custom elements',
        'shadow dom', 'html templates', 'modules', 'es modules', 'import maps'
    ],
    
    'general': []  # Catch-all for other topics
}