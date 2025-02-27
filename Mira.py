# 创建一个总是返回True的函数规则
import asyncio
import os
import re

import requests
import yaml
import aiohttp
import json
from Lib import *
from Lib.core import PluginManager, ConfigManager




logger = Logger.get_logger()

plugin_info = PluginManager.PluginInfo(
  NAME="Mira",
  AUTHOR="MMG",
  VERSION="1.0.0",
  DESCRIPTION="用于中转消息到自定义端口的mira接口插件",
  HELP_MSG="自动化插件"
)

def extract_image_url(cq_code: str) -> str:
  """
  Extracts the URL from a CQ code containing an image.

  :param cq_code: The CQ code string that contains the image URL.
  :return: The extracted URL or None if no URL is found.
  """
  # Regular expression to match the CQ:image tag and capture the URL attribute
  image_url_pattern = re.compile(r'\[CQ:image.*?url=([^,\]]+)')

  # Search for the URL within the CQ code
  match = image_url_pattern.search(cq_code)

  if match:
    # The URL is captured as the first group in the regex match
    url = match.group(1)

    # Clean up ampersand encoding (&amp; -> &)
    clean_url = url.replace("&amp;", "&")

    return clean_url
  else:
    # No URL found in the CQ code
    return None



rule = EventHandlers.CommandRule("mira", aliases={"米拉"})  # 配置基于命令的触发器和配置命令别名
reload = EventHandlers.CommandRule("reload", aliases={"配置重载"})  # 配置基于命令的触发器和配置命令别名
call = EventHandlers.CommandRule("call", aliases={"呼叫"})  # 配置基于命令的触发器和配置命令别名

callmatcher = EventHandlers.on_event(EventClassifier.GroupMessageEvent, priority=0, rules=[call])
reloadmatcher = EventHandlers.on_event(EventClassifier.GroupMessageEvent, priority=0, rules=[reload])
CMDmatcher = EventHandlers.on_event(EventClassifier.GroupMessageEvent, priority=0, rules=[rule])


@reloadmatcher.register_handler()
def reload(event_data):
  global config, target_groups
  config = load_config()
  target_groups = config.get("target_groups", "")
  llm_config = config.get("LLMserver", {})
  host = llm_config.get("host", "127.0.0.1")  # 默认主机为127.0.0.1
  port = llm_config.get("port", "")
  logger.info(f"LLM服务器配置 - 主机: {host}, 端口: {port}")
  try:
    # 使用配置的host和port
    response = requests.get(f"http://{host}:{port}/health", timeout=5)

    # 检查响应状态码
    if response.status_code == 200:
      # 尝试解析JSON响应
      try:
        status_data = response.json()
        LLMstate = "  对话服务器：" + status_data.get("status", "状态未知")
      except ValueError:
        # 如果不是JSON，使用文本响应
        LLMstate = f"  服务在线，响应: {response.text[:50]}"
    else:
      LLMstate = f"  服务返回错误状态: {response.status_code}"

  except requests.exceptions.ConnectionError:
    LLMstate = "  连接失败：服务未启动或网络问题"
  except requests.exceptions.Timeout:
    LLMstate = "  连接超时：服务响应时间过长"
  except Exception as e:
    logger.warning(f"获取服务器心跳失败: {repr(e)}")
    LLMstate = f"  检查失败: {str(e)[:50]}"

  Actions.SendMsg(
    message=QQRichText.QQRichText(
      f"Mira已重载配置\n组件状态:\n {LLMstate}"
    ), group_id=event_data["group_id"]
  ).call()
  return config, llm_config


@CMDmatcher.register_handler()
def Mira(event_data):
  Actions.SendMsg(
    message=QQRichText.QQRichText(
      "Mira已经就绪"
    ), group_id=event_data["group_id"]
  ).call()


# 读取配置文件
def load_config():
  config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "mira_config.yml")
  try:
    with open(config_path, 'r', encoding='utf-8') as f:
      config = yaml.safe_load(f)
    logger.info(f"成功加载Mira配置文件: {config_path}")
    logger.info(str(config))
    return config
  except Exception as e:
    logger.error(f"加载Mira配置文件失败: {repr(e)}")
    # 返回默认配置
    return {
      "bot_qq": ConfigManager.GlobalConfig().account.user_id,  # 如果读取失败，使用全局配置的QQ号
      "target_groups": [950173628],  # 测试群组
      "AtTrigger": True  # 默认启用@触发
    }


# 加载配置
config = load_config()

BotAt = "[CQ:at,qq=" + str(config.get("bot_qq", ""))
target_groups = config.get("target_groups", "")

"""
配置转发器触发规则
event_data.group_id
event_data.user_id
...
配置全局触发
配置群聊提条件触发
配置at触发
"""


@callmatcher.register_handler()
def Mira(event_data):
  asyncio.run(
    LLMrequest(event_data)
  )


async def LLMrequest(event_data,
                     system_prompt: list = None,
                     im: str = None
                     ) -> dict:
  llm_config = config.get("LLMserver", {})
  port = llm_config.get("port", "")
  # 设置默认系统提示
  if system_prompt is None:
    system_prompt = [
      "Analyze the user's query and select the appropriate tool to use.",
      "Provide a natural response based on the tool's output."
    ]
  url = f"http://localhost:{port}/chat"
  # 构建请求载荷
  payload = {
    "session_id": str(event_data.user_id),
    "content": str(event_data.message),
    "system_prompt": system_prompt,
  }

  # 仅当提供了图像参数时才添加
  if im is not None:
    payload["im"] = im

  # 发送请求并获取响应
  try:
    async with aiohttp.ClientSession() as session:
      async with session.post(url, json=payload) as response:
        if response.status == 200:
          Actions.SendMsg(
            message=QQRichText.QQRichText(
              str(await response.json())
            ), group_id=event_data["group_id"]
          ).call()
          return
        else:
          error_text = await response.text()
          Actions.SendMsg(
            message=QQRichText.QQRichText(
              error_text
            ), group_id=event_data["group_id"]
          ).call()
          raise Exception(f"HTTP错误 {response.status}: {error_text}")
  except Exception as e:
    raise Exception(f"LLM服务请求失败: {str(e)}")
    return


def always_match(event_data):
  if event_data.group_id in target_groups:

    return True
  else:
    return False


def at_match(event_data):
  if BotAt in event_data.message:
    print(BotAt)
    return True
  else:
    return False


# 注册事件
all_message_rule = EventHandlers.FuncRule(always_match)
at_message_rule = EventHandlers.FuncRule(at_match)
# 注册事件处理器
matcherEcho = EventHandlers.on_event(EventClassifier.GroupMessageEvent, rules=[all_message_rule])
matcherReplay = EventHandlers.on_event(EventClassifier.GroupMessageEvent, rules=[at_message_rule])
"""
配置转发器行为

"""

"""配置基于群聊id条件的触发器"""


@matcherEcho.register_handler()
def echo(event_data):
  #通过yml配置额外条件
  Echo = config.get("Echo", "")
  if Echo == True:
    message_content = str(event_data.message)
    print("code 77 " + message_content)
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        message_content
      ), group_id=event_data["group_id"]
    ).call()


"""
配置基于at的触发器
使用AtTrigger启用
"""


@matcherReplay.register_handler()
def BotAt(event_data):
  AtTrigger = config.get("AtTrigger", "")
  if AtTrigger == True:
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        "收到"
      ), group_id=event_data["group_id"]
    ).call()
