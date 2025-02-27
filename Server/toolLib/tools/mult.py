#示例：乘法器注
"""
# toolLib/tools/new_tool.py
from ..tool_configs import ToolInfo, ToolRegistry

class NewTool(ToolInfo):
    def __init__(self):
        super().__init__()
        self.NAME = "new_tool_name"
        self.DESCRIPTION = "Tool description"
        self.PARAMETERS = {
            # 参数定义
        }

    @staticmethod
    def execute(**kwargs):
        # 工具实现
        pass

# 注册工具
ToolRegistry.register_tool(NewTool)
"""
# math_tools.py
from ..tool_configs import ToolInfo, ToolRegistry

class MultiplyTool(ToolInfo):
    def __init__(self):
        super().__init__()
        self.NAME = "multiply_numbers"
        self.DESCRIPTION = "Multiply two numbers together"
        self.PARAMETERS = {
            "type": "OBJECT",
            "properties": {
                "number1": {
                    "type": "NUMBER",
                    "description": "The first number to multiply",
                },
                "number2": {
                    "type": "NUMBER",
                    "description": "The second number to multiply",
                },
            },
            "required": ["number1", "number2"]

        }

    @staticmethod
    def execute(number1: float, number2: float) -> float:
        """执行乘法运算"""
        print("code11")
        return number1 * number2

# 注册工具
ToolRegistry.register_tool(MultiplyTool)
