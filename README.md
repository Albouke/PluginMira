google genai sdk是必要的 https://github.com/googleapis/python-genai 
项目结构
Murainbot（机器人根目录）
│
├── plugin/
│   ├── mira.py                 # Mira 插件主文件
│   └── config/
│       └── mira_config.yml     # Mira 配置文件
│
└── Server/                     # LLM 服务器目录
    ├── LLMserver.py            # LLM 服务器主程序
    └── google/
        └── genai/              # Google Generative AI 库
