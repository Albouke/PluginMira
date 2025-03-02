# toolLib/tools/steam_info_tool.py
from ..tool_configs import ToolInfo, ToolRegistry
import sys
import os

# 确保能够导入 steam.py 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if root_dir not in sys.path:
  sys.path.append(root_dir)

# 导入 steam.py 中的功能
from plugins.steam import SteamAPI, process_steam_id_input


class SteamInfoTool(ToolInfo):
  def __init__(self):
    super().__init__()
    self.NAME = "get_steam_player_info"
    self.DESCRIPTION = "获取 Steam 玩家的详细信息，包括基本资料、当前游戏状态、最近玩过的游戏以及游戏库统计 要求字符串输入"
    self.PARAMETERS = {
      "type": "OBJECT",
      "properties": {
        "steam_id": {
          "type": "STRING",
          "description": "玩家的 Steam 标识符，可以是 Steam 64 ID、好友代码或自定义 URL",
        },
      },
      "required": ["steam_id"]
    }

  @staticmethod
  def execute(steam_id) -> str:
    print ("code 333" + str(steam_id))
    """
    查询并返回 Steam 玩家的详细信息

    参数:
        steam_id: Steam 标识符 (Steam 64 ID、好友代码或自定义 URL)

    返回:
        格式化的玩家信息字符串
    """
    try:
      # Steam API 密钥 (使用 steam.py 中的相同密钥)
      API_KEY = "E346980886DC7CA15CC9B5E87A4109B4"

      # 创建 SteamAPI 实例
      steam_api = SteamAPI(API_KEY)

      # 处理各种格式的输入 ID，转换为标准的 Steam 64 ID
      processed_id = process_steam_id_input(steam_id, API_KEY)

      if not processed_id:
        return "无法处理提供的 Steam ID。请确保提供有效的 Steam 64 ID、好友代码或自定义 URL。"

      # 获取格式化的玩家信息
      player_info = steam_api.format_player_info(processed_id)

      if not player_info:
        return "无法获取玩家信息，该账号可能不存在或设为私密。"

      return str(player_info)

    except Exception as e:
      return f"查询 Steam 玩家信息时出错: {str(e)}"


# 注册工具
ToolRegistry.register_tool(SteamInfoTool)
