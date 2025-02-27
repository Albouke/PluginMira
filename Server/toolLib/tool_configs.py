# tool_configs.py
from typing import Dict, Any, List

class ToolInfo:
    def __init__(self):
        self.NAME: str = ""
        self.DESCRIPTION: str = ""
        self.PARAMETERS: Dict[str, Any] = {
            "type": "OBJECT",
            "properties": {},
            "required": []
        }

    def get_config(self) -> Dict[str, Any]:
        """返回工具配置"""
        return {
            "name": self.NAME,
            "description": self.DESCRIPTION,
            "parameters": self.PARAMETERS
        }

class ToolRegistry:
    _instance = None
    _tools: Dict[str, Any] = {}
    _configs: Dict[str, Dict] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_tool(cls, tool_class):
        """注册工具类"""
        tool_info = tool_class()
        cls._tools[tool_info.NAME] = tool_class
        cls._configs[tool_info.NAME] = tool_info.get_config()

    @classmethod
    def get_tools(cls) -> Dict[str, Any]:
        """获取所有工具"""
        return cls._tools

    @classmethod
    def get_configs(cls) -> List[Dict]:
        """获取所有配置"""
        return list(cls._configs.values())
