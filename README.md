google genai sdk是必要的 https://github.com/googleapis/python-genai 

***插件功能简介***
Mira 是一个为MurainBot框架提供接入Gemini插件 ~~附带独立的FastAPI服务组件~~，支持：
工具调用和自动工具导入
session会话管理
Gemini Vision 图像理解

<details>
<summary>目录结构</summary>

```
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
```

</details>
