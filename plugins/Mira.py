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


# 注册各种命令规则
rule = EventHandlers.CommandRule("mira", aliases={"米拉"})
reload = EventHandlers.CommandRule("reload", aliases={"配置重载"})
call = EventHandlers.CommandRule("call", aliases={"呼叫"})
visible = EventHandlers.CommandRule("visible", aliases={"思维可见"})
model_cmd = EventHandlers.CommandRule("model", aliases={"模型"})
clean_cmd = EventHandlers.CommandRule("clean", aliases={"清空历史记录"})
prompt_cmd = EventHandlers.CommandRule("prompt", aliases={"提示设置"})
preprompt_cmd = EventHandlers.CommandRule("preprompt", aliases={"人格"})

# 设置相应的匹配器
callmatcher = EventHandlers.on_event(EventClassifier.GroupMessageEvent, priority=0, rules=[call])
reloadmatcher = EventHandlers.on_event(EventClassifier.GroupMessageEvent, priority=0, rules=[reload])
CMDmatcher = EventHandlers.on_event(EventClassifier.GroupMessageEvent, priority=0, rules=[rule])
visiblematcher = EventHandlers.on_event(EventClassifier.GroupMessageEvent, priority=0, rules=[visible])
modelmatcher = EventHandlers.on_event(EventClassifier.GroupMessageEvent, priority=0, rules=[model_cmd])
cleanmatcher = EventHandlers.on_event(EventClassifier.GroupMessageEvent, priority=0, rules=[clean_cmd])
promptmatcher = EventHandlers.on_event(EventClassifier.GroupMessageEvent, priority=0, rules=[prompt_cmd])
prepromptmatcher = EventHandlers.on_event(EventClassifier.GroupMessageEvent, priority=0, rules=[preprompt_cmd])


def clean_session(session_id, server_url="http://localhost:5888"):
  """
  发送请求清理特定会话的对话历史

  Args:
      session_id: 要清理的会话ID
      server_url: 服务器URL

  Returns:
      dict: 服务器响应
  """
  endpoint = f"{server_url}/clean-session"
  data = {"session_id": session_id}

  response = requests.post(endpoint, json=data)

  if response.status_code == 200:
    return response.json()
  else:
    print(f"Error: {response.status_code}")
    print(response.text)
    return None


@cleanmatcher.register_handler()
def clean(event_data):
  result = clean_session(str(event_data.user_id))
  Actions.SendMsg(
    message=QQRichText.QQRichText(
      f"{result}"
    ), group_id=event_data["group_id"]
  ).call()


@reloadmatcher.register_handler()
def reload(event_data):
  global config, target_groups, current_model
  config = load_config()
  target_groups = config.get("target_groups", "")
  llm_config = config.get("LLMserver", {})
  host = llm_config.get("host", "127.0.0.1")  # 默认主机为127.0.0.1
  port = llm_config.get("port", "")

  # 从配置读取默认模型
  current_model = config.get("model", "gemini")

  logger.info(f"LLM服务器配置 - 主机: {host}, 端口: {port}, 模型: {current_model}")
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
      f"Mira已重载配置\n组件状态:\n {LLMstate}\n当前模型: {current_model}\n思维可见: {'开启' if think_visible else '关闭'}"
    ), group_id=event_data["group_id"]
  ).call()
  return config, llm_config


@CMDmatcher.register_handler()
def Mira(event_data):
  Actions.SendMsg(
    message=QQRichText.QQRichText(
      f"Mira已经就绪\n当前模型: {current_model}\n思维可见: {'开启' if think_visible else '关闭'}"
    ), group_id=event_data["group_id"]
  ).call()


@visiblematcher.register_handler()
def toggle_visibility(event_data):
  global think_visible
  think_visible = not think_visible
  status = "开启" if think_visible else "关闭"
  Actions.SendMsg(
    message=QQRichText.QQRichText(
      f"思维可见已{status}，当前模型: {current_model}"
    ), group_id=event_data["group_id"]
  ).call()


@promptmatcher.register_handler()
def change_model(event_data):
  global current_prompt
  # 从命令中提取提示
  message = str(event_data.message).strip()
  print("code 77 " + message)

  # 使用正则表达式提取模型名称 - 匹配 prompt|提示设置 #modelname# 格式
  match = re.search(r'(?:prompt|提示设置)\s*#?([\w\u4e00-\u9fff]+)#?', message, re.IGNORECASE)
  if match:
    new_prompt = match.group(1).lower()
    current_prompt = new_prompt
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        f"模型prompt已更改"
      ), group_id=event_data["group_id"]
    ).call()


@prepromptmatcher.register_handler()
def prompt(event_data):
  print("code 72")
  global current_prompt, promptLib
  # 从命令中提取提示
  message = str(event_data.message).strip()
  print("code 77 " + message)
  match = re.search(r'(?:preprompt|人格)\s*#?([a-zA-Z0-9]+)#?', message, re.IGNORECASE)
  if match:
    new_model = match.group(1)
    print(new_model)
    if new_model in promptLib:
      current_prompt = promptLib.get(f"{new_model}", "")
      Actions.SendMsg(
        message=QQRichText.QQRichText(
          f"{new_model}已连线"
        ), group_id=event_data["group_id"]
      ).call()


@modelmatcher.register_handler()
def change_model(event_data):
  global current_model
  # 从命令中提取模型名称
  message = str(event_data.message).strip()

  # 使用正则表达式提取模型名称 - 匹配 /model {modelname} 或 /模型 {modelname} 格式
  match = re.search(r'[/]?(?:model|模型)\s*[{]?([a-zA-Z0-9]+)[}]?', message, re.IGNORECASE)
  if match:
    new_model = match.group(1).lower()

    # 验证模型名称
    supported_models = ["gemini", "deepseek","hybrid"]
    if new_model in supported_models:
      current_model = new_model
      if current_model == "hybrid":
        Actions.SendMsg(
          message=QQRichText.QQRichText(
            f"已切换至hybrid 混合模式，思维可见: {'开启' if think_visible else '关闭'}"
          ), group_id=event_data["group_id"]
        ).call()
      else:
        Actions.SendMsg(
          message=QQRichText.QQRichText(
            f"已切换至 {new_model} 模型，思维可见: {'开启' if think_visible else '关闭'}"
          ), group_id=event_data["group_id"]
        ).call()

    else:
      supported_list = ", ".join(supported_models)
      Actions.SendMsg(
        message=QQRichText.QQRichText(
          f"不支持的模型: {new_model}，目前支持的模型有: {supported_list}"
        ), group_id=event_data["group_id"]
      ).call()
  else:
    supported_models = ["gemini", "deepseek"]
    supported_list = ", ".join(supported_models)
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        f"未知命令，目前支持的模型有: {supported_list}"
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
      "AtTrigger": True,  # 默认启用@触发
      "model": "gemini",  # 默认模型
      "ThinkVisible": False  # 默认不显示思考过程
    }


# 加载配置
config = load_config()

# 初始化全局变量
BotAt = "[CQ:at,qq=" + str(config.get("bot_qq", ""))
target_groups = config.get("target_groups", "")
current_model = config.get("model", "gemini")  # 从配置加载默认模型
think_visible = config.get("ThinkVisible", False)  # 从配置加载默认思考可见性
promptLib = config.get("prompt", {"Mira": ""})  # 从配置加载默认思考可见性
current_prompt = promptLib.get("Mira", "")


@callmatcher.register_handler()
def call_llm(event_data):
  asyncio.run(
    LLMrequest(event_data)
  )


async def LLMrequest(event_data,
                     system_prompt: list = None,
                     im: str = None) -> dict:
  global current_model, think_visible, current_prompt
  llm_config = config.get("LLMserver", {})
  port = llm_config.get("port", "")
  toolprompt="你是一个可以使用工具的助手，你拥有两个工具，包括乘法计算机和查询steam的工具，用户可能会在对话里提到关于库存，游戏资料，主页等涉及的steam，在他们讨论相关话题并提供类似steam 64id或好友代码的字符串，一旦在对话里出现非计算用的长数字字符串，你***必须尝试使用你的get_steam_player_info进行查询***，工具会自己判断这是不是有效的id,相信steam工具查询返回的背景信息以了解用户"

  # 设置默认系统提示，根据不同模型有不同的默认值
  if system_prompt is None:
    if current_model == "gemini":
      system_prompt = [
        toolprompt,
        current_prompt
      ]
    elif current_model == "deepseek":
      system_prompt = [
        "",  # 占位，不使用
        current_prompt
      ]
    else:
      # 其他模型的默认提示
      system_prompt = [
        toolprompt,
        current_prompt
      ]

  # 根据模型类型构建URL
  url = f"http://localhost:{port}/{current_model}/chat"

  # 构建请求载荷
  payload = {
    "session_id": str(event_data.user_id),
    "content": str(event_data.message),
    "system_prompt": system_prompt,
  }

  # 仅当提供了图像参数时才添加，且仅Gemini支持图像
  if im is not None and current_model == "gemini":
    payload["im"] = im

  # 发送请求并获取响应
  try:
    async with aiohttp.ClientSession() as session:
      async with session.post(url, json=payload) as response:
        if response.status == 200:
          response_data = await response.json()

          # 处理响应内容
          if isinstance(response_data, dict):
            message_text = response_data.get("response", "")

            # 如果是DeepSeek且思考可见性开启
            if current_model == "deepseek" and think_visible and "reasoning" in response_data:
              reasoning_text = response_data.get("reasoning", "")
              full_response = f"思考过程:\n{reasoning_text}\n\n回答:\n{message_text}"
              Actions.SendMsg(
                message=QQRichText.QQRichText(full_response),
                group_id=event_data["group_id"]
              ).call()
            else:
              # 普通响应
              Actions.SendMsg(
                message=QQRichText.QQRichText(message_text),
                group_id=event_data["group_id"]
              ).call()
          else:
            # 如果返回的不是字典格式
            Actions.SendMsg(
              message=QQRichText.QQRichText(
                str(response_data)
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
    error_msg = f"LLM服务请求失败: {str(e)}"
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        error_msg
      ), group_id=event_data["group_id"]
    ).call()
    raise Exception(error_msg)
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


@matcherEcho.register_handler()
def echo(event_data):
  # 通过yml配置额外条件
  Echo = config.get("Echo", "")
  if Echo == True:
    message_content = str(event_data.message)
    print("code 77 " + message_content)
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        message_content
      ), group_id=event_data["group_id"]
    ).call()


@matcherReplay.register_handler()
def BotAt(event_data):
  AtTrigger = config.get("AtTrigger", "")
  if AtTrigger == True:
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        "收到"
      ), group_id=event_data["group_id"]
    ).call()
