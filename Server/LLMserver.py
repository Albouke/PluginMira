import os
import time
import asyncio
from io import BytesIO
from PIL import Image
import importlib
from pathlib import Path
from typing import Optional, Any, List, Dict, Tuple, Union
import json
import redis
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel
from toolLib.tool_configs import ToolRegistry
from openai import OpenAI
import yaml
from datetime import datetime
import logging
import concurrent.futures

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLMserver")


def load_config():
  config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "server_config.yml")
  with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
  return config


# Global config
config = load_config()


# Message Queue Service - Handles all state management
class MessageQueueService:
  def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, redis_db: int = 0):
    self.redis = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
    self.message_ttl = 24 * 60 * 60  # 24 hours
    self.lock_ttl = 5 * 60  # 5 minutes max lock time to prevent deadlocks

  def get_messages(self, session_id: str) -> List[Dict[str, str]]:
    messages_str = self.redis.get(f"chat:{session_id}")
    if messages_str:
      return json.loads(messages_str)
    return []

  def add_message(self, session_id: str, role: str, content: str, history_limit: int):
    messages = self.get_messages(session_id)
    messages.append({"role": role, "content": content})

    # Keep only the most recent messages
    if len(messages) > history_limit:
      messages = messages[-history_limit:]

    self.redis.setex(
      f"chat:{session_id}",
      self.message_ttl,
      json.dumps(messages)
    )

  def format_messages(self, messages: List[Dict[str, str]]) -> str:
    formatted = []
    for msg in messages:
      role_prefix = "User" if msg["role"] == "user" else "Assistant"
      formatted.append(f"{role_prefix}: {msg['content']}")
    return "\n".join(formatted)

  def clear_session(self, session_id: str) -> bool:
    chat_key = f"chat:{session_id}"
    system_prompt_key = f"chat:{session_id}:has_system_prompt"
    lock_key = f"chat:{session_id}:lock"
    lock_time_key = f"chat:{session_id}:lock_time"

    pipe = self.redis.pipeline()
    pipe.delete(chat_key)
    pipe.delete(system_prompt_key)
    pipe.delete(lock_key)
    pipe.delete(lock_time_key)
    results = pipe.execute()

    return any(result == 1 for result in results)

  def prepare_deepseek_messages(self, messages: List[Dict[str, str]], system_prompt: str = None) -> List[
    Dict[str, str]]:
    if not messages:
      return []

    # Normalize role names
    normalized_messages = []
    for msg in messages:
      role = msg["role"].lower()
      if role in ["user", "assistant"]:
        normalized_messages.append({"role": role, "content": msg["content"]})

    if not normalized_messages:
      return []

    # Ensure messages start with user
    if normalized_messages[0]["role"] != "user":
      normalized_messages = normalized_messages[1:]
      if not normalized_messages:
        return []

    # Create strictly alternating message sequence
    formatted_messages = []
    current_role = "user"

    for i, msg in enumerate(normalized_messages):
      if msg["role"] == current_role:
        # If first user message and system prompt exists, embed it
        if i == 0 and current_role == "user" and system_prompt:
          formatted_messages.append({
            "role": "user",
            "content": f"{system_prompt}\n\n{msg['content']}"
          })
        else:
          formatted_messages.append(msg)

        # Switch expected next role
        current_role = "assistant" if current_role == "user" else "user"
      else:
        # Skip messages that don't match expected order
        continue

    # Ensure messages end with user input
    if formatted_messages and formatted_messages[-1]["role"] == "assistant":
      formatted_messages = formatted_messages[:-1]

    return formatted_messages

  def has_system_prompt_embedded(self, session_id: str) -> bool:
    key = f"chat:{session_id}:has_system_prompt"
    return bool(self.redis.get(key))

  def mark_system_prompt_embedded(self, session_id: str):
    self.redis.setex(
      f"chat:{session_id}:has_system_prompt",
      self.message_ttl,
      "1"
    )

  # 会话锁定相关方法
  def lock_session(self, session_id: str) -> bool:
    """
    尝试锁定会话。如果会话已被锁定，返回False；否则锁定会话并返回True
    """
    lock_key = f"chat:{session_id}:lock"
    # 使用setnx原子操作尝试获取锁
    lock_acquired = self.redis.setnx(lock_key, "1")

    if lock_acquired:
      # 设置锁的过期时间，防止死锁
      self.redis.expire(lock_key, self.lock_ttl)
      # 记录锁定时间
      self.redis.set(f"chat:{session_id}:lock_time", str(time.time()))
      logger.info(f"Session {session_id} locked successfully")
      return True

    logger.info(f"Failed to lock session {session_id} - already locked")
    return False

  def unlock_session(self, session_id: str):
    """
    解锁会话，允许处理新的请求
    """
    lock_key = f"chat:{session_id}:lock"
    result = self.redis.delete(lock_key)
    logger.info(f"Unlocked session {session_id}, result: {result}")

  def get_lock_duration(self, session_id: str) -> Tuple[bool, float]:
    """
    检查会话是否被锁定，并返回锁定持续时间
    返回: (是否锁定, 锁定时长秒数)
    """
    lock_key = f"chat:{session_id}:lock"
    lock_time_key = f"chat:{session_id}:lock_time"

    is_locked = bool(self.redis.exists(lock_key))
    if not is_locked:
      return False, 0

    lock_time_str = self.redis.get(lock_time_key)
    if not lock_time_str:
      # 如果有锁但没有时间记录，可能是异常情况
      return True, 0

    try:
      lock_time = float(lock_time_str)
      duration = time.time() - lock_time
      return True, duration
    except (ValueError, TypeError):
      return True, 0


async def process_gemini_query(contents: Union[str, List], sys_prompt1: str, sys_prompt2: str,
                               temperature: float = 0.7) -> str:
  """处理Gemini API查询的函数，支持工具调用和多模态输入"""
  api_key = config.get("gemini", "")
  client = genai.Client(api_key=api_key)
  loop = asyncio.get_event_loop()

  # 加载工具
  tools_dir = Path("toolLib/tools")
  for tool_file in tools_dir.glob("*.py"):
    if not tool_file.stem.startswith("_"):
      importlib.import_module(f"toolLib.tools.{tool_file.stem}")

  # 打印已注册的所有工具
  logger.info(f"已注册工具: {[tool['name'] for tool in ToolRegistry.get_configs()]}")

  # 配置安全设置
  safety_settings = [
    types.SafetySetting(
      category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
      threshold='BLOCK_NONE',
    )
  ]

  # 工具检测配置
  tool_detection_config = types.GenerateContentConfig(
    safety_settings=safety_settings,
    temperature=temperature,
    system_instruction=sys_prompt1,
    tools=[types.Tool(function_declarations=ToolRegistry.get_configs())]
  )

  # 最终回答配置
  final_response_config = types.GenerateContentConfig(
    safety_settings=safety_settings,
    temperature=temperature,
    system_instruction=sys_prompt2,
  )

  try:
    # 第一步：检测是否需要工具
    logger.info("Step 1: Checking if tool usage is needed")
    with concurrent.futures.ThreadPoolExecutor() as pool:
      first_response = await loop.run_in_executor(
        pool,
        lambda: client.models.generate_content(
          model='gemini-2.0-flash-exp',
          contents=contents,
          config=tool_detection_config
        )
      )

    # 详细日志：响应结构检查
    logger.info(f"响应类型: {type(first_response)}")
    logger.info(f"候选项数量: {len(first_response.candidates)}")
    logger.info(f"第一个候选项内容部分数量: {len(first_response.candidates[0].content.parts)}")
    logger.info(f"第一个部分类型: {type(first_response.candidates[0].content.parts[0])}")

    # 获取第一部分
    first_part = first_response.candidates[0].content.parts[0]

    # 检查是否有function_call属性
    logger.info(f"first_part属性列表: {dir(first_part)}")
    logger.info(f"是否有function_call属性: {hasattr(first_part, 'function_call')}")

    if hasattr(first_part, 'function_call'):
      # 检查function_call是否为None
      function_call = first_part.function_call
      logger.info(f"function_call值: {function_call}")
      logger.info(f"function_call类型: {type(function_call)}")

      if function_call is not None:
        # 查看function_call的所有属性
        logger.info(f"function_call属性列表: {dir(function_call)}")

        # 检查是否有name属性
        if hasattr(function_call, 'name'):
          logger.info(f"函数名称: {function_call.name}")
        else:
          logger.warning("function_call没有name属性")

        # 检查是否有args属性
        if hasattr(function_call, 'args'):
          logger.info(f"函数参数: {function_call.args}")
          logger.info(f"函数参数类型: {type(function_call.args)}")

          # 详细记录每个参数
          if isinstance(function_call.args, dict):
            for k, v in function_call.args.items():
              logger.info(f"参数 {k}: {v} (类型: {type(v)})")
        else:
          logger.warning("function_call没有args属性")
      else:
        logger.warning("function_call属性存在但值为None")

    # 使用更健壮的检查方式
    uses_tool = (hasattr(first_part, 'function_call') and
                 first_part.function_call is not None and
                 hasattr(first_part.function_call, 'name') and
                 first_part.function_call.name is not None)

    logger.info(f"是否使用工具: {uses_tool}")

    # 如果不使用工具，直接使用sys_prompt2获取最终回答
    if not uses_tool:
      logger.info("不需要工具，生成直接响应")
      with concurrent.futures.ThreadPoolExecutor() as pool:
        final_response = await loop.run_in_executor(
          pool,
          lambda: client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=contents,
            config=final_response_config
          )
        )
      return final_response.text

    # 使用工具处理
    logger.info("工具使用已检测，处理工具请求")
    function_name = first_part.function_call.name
    function_args = first_part.function_call.args
    logger.info(f"将执行函数: {function_name}，参数: {function_args}")

    # 执行工具函数
    tools = ToolRegistry.get_tools()
    if function_name not in tools:
      logger.error(f"函数 {function_name} 未在注册表中找到")
      raise ValueError(f"Function {function_name} not found in registry")

    tool_class = tools[function_name]
    logger.info(f"执行工具类: {tool_class.__name__}")

    with concurrent.futures.ThreadPoolExecutor() as pool:
      function_result = await loop.run_in_executor(
        pool,
        lambda: tool_class.execute(**function_args)
      )

    logger.info(f"工具执行完成: {function_name}, 结果类型: {type(function_result)}")
    logger.info(f"结果预览: {str(function_result)[:200]}...")

    # 构建会话历史
    user_message = {
      "role": "user",
      "parts": [{"text": contents if isinstance(contents, str) else contents[0]}]
    }

    if isinstance(contents, list) and len(contents) > 1:
      user_message["parts"].append({"inline_data": contents[1]})

    model_function_call = {
      "role": "model",
      "parts": [first_part]
    }

    function_response_message = {
      "role": "function",
      "parts": [{
        "functionResponse": {
          "name": function_name,
          "response": {"result": function_result}
        }
      }]
    }

    conversation_history = [
      user_message,
      model_function_call,
      function_response_message
    ]

    # 生成最终响应
    logger.info("生成带有正确结构化对话历史的最终响应")
    with concurrent.futures.ThreadPoolExecutor() as pool:
      final_response = await loop.run_in_executor(
        pool,
        lambda: client.models.generate_content(
          model='gemini-2.0-flash-exp',
          contents=conversation_history,
          config=final_response_config
        )
      )

    return final_response.text

  except Exception as e:
    logger.error(f"Gemini API处理错误: {str(e)}")
    # 打印完整堆栈跟踪
    import traceback
    logger.error(f"堆栈跟踪: {traceback.format_exc()}")

    # 尝试回退到直接响应
    try:
      logger.info("回退到不使用工具的直接响应")
      with concurrent.futures.ThreadPoolExecutor() as pool:
        fallback_response = await loop.run_in_executor(
          pool,
          lambda: client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=contents,
            config=final_response_config
          )
        )
      return fallback_response.text
    except Exception as inner_e:
      raise Exception(f"查询处理错误: {str(e)}. 回退也失败: {str(inner_e)}")
# Stateless LLM API clients
"""
class GeminiAPI:
  @staticmethod
  async def process_query(query: Any, sys_prompt1: str, sys_prompt2: str, temperature: float = 0.7) -> str:
    api_key = config.get("gemini", "")
    client = genai.Client(api_key=api_key)

    # Load tools
    tools_dir = Path("toolLib/tools")
    for tool_file in tools_dir.glob("*.py"):
      if not tool_file.stem.startswith("_"):
        importlib.import_module(f"toolLib.tools.{tool_file.stem}")

    # Create config
    safety_settings = [
      types.SafetySetting(
        category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
        threshold='BLOCK_NONE',
      )
    ]

    config1 = types.GenerateContentConfig(
      safety_settings=safety_settings,
      temperature=temperature,
      system_instruction=sys_prompt1,
      tools=[types.Tool(function_declarations=ToolRegistry.get_configs())]
    )

    config2 = types.GenerateContentConfig(
      safety_settings=safety_settings,
      temperature=temperature,
      system_instruction=sys_prompt2,
      tools=[types.Tool(function_declarations=ToolRegistry.get_configs())]
    )

    try:
      # Standardize query format
      if not isinstance(query, (list, tuple)):
        query = [query]

      # First call to check for function calls - use loop.run_in_executor for potentially blocking operations
      loop = asyncio.get_event_loop()
      with concurrent.futures.ThreadPoolExecutor() as pool:
        first_response = await loop.run_in_executor(
          pool,
          lambda: client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=query,
            config=config1
          )
        )

      first_part = first_response.candidates[0].content.parts[0]

      # No function call, use sys_prompt2 directly
      if not hasattr(first_part, 'function_call'):
        with concurrent.futures.ThreadPoolExecutor() as pool:
          final_response = await loop.run_in_executor(
            pool,
            lambda: client.models.generate_content(
              model='gemini-2.0-flash-exp',
              contents=query,
              config=config2
            )
          )
        return final_response.text

      # With function call
      function_name = first_part.function_call.name
      function_args = first_part.function_call.args

      # Execute tool function
      tools = ToolRegistry.get_tools()
      if function_name not in tools:
        raise ValueError(f"Function {function_name} not found")

      tool_class = tools[function_name]
      with concurrent.futures.ThreadPoolExecutor() as pool:
        function_response = await loop.run_in_executor(
          pool,
          lambda: tool_class.execute(**function_args)
        )

      # Create function response
      function_response_part = types.Part.from_function_response(
        name=function_name,
        response={'result': function_response}
      )

      # Final call with function results
      with concurrent.futures.ThreadPoolExecutor() as pool:
        final_response = await loop.run_in_executor(
          pool,
          lambda: client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=[
              types.Part.from_text(query),
              first_part,
              function_response_part,
            ],
            config=config2
          )
        )

      return final_response.text

    except Exception as e:
      # Fallback to direct response
      try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
          response = await loop.run_in_executor(
            pool,
            lambda: client.models.generate_content(
              model='gemini-2.0-flash-exp',
              contents=[query],
              config=config2
            )
          )
        return response.text
      except Exception as inner_e:
        raise Exception(f"Error processing query: {str(e)}")
"""




class DeepSeekAPI:
  @staticmethod
  async def process_query(messages: List[Dict[str, str]], temperature: float = 0.7) -> Dict[str, str]:
    """Stateless function to process queries with DeepSeek API"""
    api_key = config.get("deepseek", "")
    if not api_key:
      raise ValueError("DeepSeek API key not found in config")

    client = OpenAI(
      api_key=api_key,
      base_url="https://api.deepseek.com"
    )

    try:
      # Prepare request parameters
      params = {
        "model": "deepseek-reasoner",
        "messages": messages,
        "temperature": temperature
      }

      # Send request using asyncio to prevent blocking
      loop = asyncio.get_event_loop()
      with concurrent.futures.ThreadPoolExecutor() as pool:
        response = await loop.run_in_executor(
          pool,
          lambda: client.chat.completions.create(**params)
        )

      # Extract results
      reasoning_content = response.choices[0].message.reasoning_content
      content = response.choices[0].message.content

      return {
        "reasoning_content": reasoning_content,
        "content": content
      }

    except Exception as e:
      raise Exception(f"DeepSeek API request failed: {str(e)}")


# FastAPI app definition
app = FastAPI()


# Request models
class ChatRequest(BaseModel):
  session_id: str
  content: str
  system_prompt: List[str]
  im: Optional[str] = None
  history_limit: int = 10


class ChatResponse(BaseModel):
  response: str
  reasoning: Optional[str] = None
  status: str = "success"
  processing_time: Optional[str] = None


class CleanSessionRequest(BaseModel):
  session_id: str


class CleanSessionResponse(BaseModel):
  status: str
  message: str


# 格式化处理时间
def format_duration(seconds: float) -> str:
  """将秒数格式化为 mm:ss:msms"""
  minutes = int(seconds // 60)
  seconds_remainder = seconds % 60
  milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
  return f"{minutes:02d}:{int(seconds_remainder):02d}:{milliseconds:03d}"


# 创建线程池执行器，用于异步执行API调用
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=30)


# API endpoints - 同步处理请求
@app.post("/gemini/chat", response_model=ChatResponse)
async def gemini_chat_endpoint(request: ChatRequest):
  # Initialize message queue service
  message_service = MessageQueueService(redis_host='localhost', redis_port=6379, redis_db=0)
  lock_acquired = False

  try:
    if len(request.system_prompt) != 2:
      raise HTTPException(
        status_code=400,
        detail="system_prompt must contain exactly 2 elements"
      )

    # 检查会话锁状态
    is_locked, duration = message_service.get_lock_duration(request.session_id)
    if is_locked:
      # 会话已锁定，返回处理中状态
      formatted_duration = format_duration(duration)
      logger.info(
        f"Session {request.session_id} is locked. Returning processing status. Duration: {formatted_duration}")
      return ChatResponse(
        response=f"Mira正在处理中，已耗时：{formatted_duration}",
        status="processing",
        processing_time=formatted_duration
      )

    # 尝试获取锁
    lock_acquired = message_service.lock_session(request.session_id)
    if not lock_acquired:
      # 锁定失败，再次检查状态
      is_locked, duration = message_service.get_lock_duration(request.session_id)
      formatted_duration = format_duration(duration)
      logger.info(
        f"Failed to acquire lock for session {request.session_id}. Returning processing status. Duration: {formatted_duration}")
      return ChatResponse(
        response=f"Mira正在处理中，已耗时：{formatted_duration}",
        status="processing",
        processing_time=formatted_duration
      )

    start_time = time.time()
    logger.info(f"Processing request for session {request.session_id}")

    # 添加用户消息到历史
    message_service.add_message(
      request.session_id,
      "user",
      request.content,
      request.history_limit
    )

    # 获取历史消息
    messages = message_service.get_messages(request.session_id)
    formatted_history = message_service.format_messages(messages)

    # 构建查询文本
    text_query = f"Chat session_id:{request.session_id}\nConversation history:\n{formatted_history}\nCurrent query: {request.content}"

    # 准备内容 - 根据是否有图片决定内容格式
    if request.im:
      img = Image.open(BytesIO(open(request.im, "rb").read()))
      contents = [text_query, img]
    else:
      contents = text_query  # 纯文本查询直接使用字符串

    # 调用重构后的Gemini API
    logger.info(f"Calling Gemini API for session {request.session_id}")
    result = await process_gemini_query(
      contents=contents,
      sys_prompt1=request.system_prompt[0],
      sys_prompt2=request.system_prompt[1]
    )

    # 计算处理时间
    elapsed_time = time.time() - start_time
    formatted_time = format_duration(elapsed_time)
    logger.info(f"Request for session {request.session_id} processed in {formatted_time}")

    # 添加助手响应到消息池
    message_service.add_message(
      request.session_id,
      "assistant",
      result,
      request.history_limit
    )

    # 返回处理结果
    return ChatResponse(
      response=result,
      status="success",
      processing_time=formatted_time
    )

  except Exception as e:
    logger.error(f"Error processing request for session {request.session_id}: {str(e)}")
    raise HTTPException(
      status_code=500,
      detail=f"Error processing request: {str(e)}"
    )
  finally:
    # 确保释放锁
    if lock_acquired:
      message_service.unlock_session(request.session_id)
      logger.info(f"Session {request.session_id} unlocked after processing")


@app.post("/deepseek/chat", response_model=ChatResponse)
async def deepseek_chat_endpoint(request: ChatRequest):
  # Initialize message queue service
  message_service = MessageQueueService(redis_host='localhost', redis_port=6379, redis_db=0)
  lock_acquired = False

  try:
    # 检查会话锁状态
    is_locked, duration = message_service.get_lock_duration(request.session_id)
    if is_locked:
      # 会话已锁定，返回处理中状态
      formatted_duration = format_duration(duration)
      logger.info(
        f"Session {request.session_id} is locked. Returning processing status. Duration: {formatted_duration}")
      return ChatResponse(
        response=f"Mira正在处理中，已耗时：{formatted_duration}",
        status="processing",
        processing_time=formatted_duration
      )

    # 尝试获取锁
    lock_acquired = message_service.lock_session(request.session_id)
    if not lock_acquired:
      # 锁定失败，再次检查状态
      is_locked, duration = message_service.get_lock_duration(request.session_id)
      formatted_duration = format_duration(duration)
      logger.info(
        f"Failed to acquire lock for session {request.session_id}. Returning processing status. Duration: {formatted_duration}")
      return ChatResponse(
        response=f"Mira正在处理中，已耗时：{formatted_duration}",
        status="processing",
        processing_time=formatted_duration
      )

    start_time = time.time()
    logger.info(f"Processing DeepSeek request for session {request.session_id}")

    # 添加用户消息到历史
    message_service.add_message(
      request.session_id,
      "user",
      request.content,
      request.history_limit
    )

    # 获取历史消息
    raw_messages = message_service.get_messages(request.session_id)

    # 检查是否需要嵌入系统提示
    system_prompt = None
    if not message_service.has_system_prompt_embedded(request.session_id):
      system_prompt = request.system_prompt[1] if len(request.system_prompt) > 1 else None
      if system_prompt:
        message_service.mark_system_prompt_embedded(request.session_id)

    # 准备DeepSeek的消息格式
    deepseek_messages = message_service.prepare_deepseek_messages(raw_messages, system_prompt)

    if not deepseek_messages:
      raise ValueError("No valid messages to send to DeepSeek API")

    # 调用DeepSeek API
    logger.info(f"Calling DeepSeek API for session {request.session_id}")
    result = await DeepSeekAPI.process_query(
      messages=deepseek_messages,
      temperature=0.7
    )

    # 计算处理时间
    elapsed_time = time.time() - start_time
    formatted_time = format_duration(elapsed_time)
    logger.info(f"DeepSeek request for session {request.session_id} processed in {formatted_time}")

    # 提取结果
    reasoning = result.get("reasoning_content", "")
    content = result.get("content", "")

    # 添加助手响应到消息池
    message_service.add_message(
      request.session_id,
      "assistant",
      content,
      request.history_limit
    )

    # 返回处理结果
    return ChatResponse(
      response=content,
      reasoning=reasoning,
      status="success",
      processing_time=formatted_time
    )

  except Exception as e:
    logger.error(f"Error processing DeepSeek request for session {request.session_id}: {str(e)}")
    raise HTTPException(
      status_code=500,
      detail=f"Error processing DeepSeek request: {str(e)}"
    )
  finally:
    # 确保释放锁
    if lock_acquired:
      message_service.unlock_session(request.session_id)
      logger.info(f"Session {request.session_id} unlocked after DeepSeek processing")


@app.post("/hybrid/chat", response_model=ChatResponse)
async def gemini_deepseek_chat_endpoint(request: ChatRequest):
  # Initialize message queue service
  message_service = MessageQueueService(redis_host='localhost', redis_port=6379, redis_db=0)
  lock_acquired = False

  try:
    if len(request.system_prompt) != 2:
      raise HTTPException(
        status_code=400,
        detail="system_prompt must contain exactly 2 elements"
      )

    # Check session lock status
    is_locked, duration = message_service.get_lock_duration(request.session_id)
    if is_locked:
      formatted_duration = format_duration(duration)
      logger.info(
        f"Session {request.session_id} is locked. Returning processing status. Duration: {formatted_duration}")
      return ChatResponse(
        response=f"Mira正在处理中，已耗时：{formatted_duration}",
        status="processing",
        processing_time=formatted_duration
      )

    # Try to acquire lock
    lock_acquired = message_service.lock_session(request.session_id)
    if not lock_acquired:
      is_locked, duration = message_service.get_lock_duration(request.session_id)
      formatted_duration = format_duration(duration)
      logger.info(
        f"Failed to acquire lock for session {request.session_id}. Returning processing status. Duration: {formatted_duration}")
      return ChatResponse(
        response=f"Mira正在处理中，已耗时：{formatted_duration}",
        status="processing",
        processing_time=formatted_duration
      )

    start_time = time.time()
    logger.info(f"Processing hybrid request for session {request.session_id}")

    # Add user message to history ONLY AFTER we've confirmed we can process it
    message_service.add_message(
      request.session_id,
      "user",
      request.content,
      request.history_limit
    )

    # Get history messages
    messages = message_service.get_messages(request.session_id)
    formatted_history = message_service.format_messages(messages)

    # Build query text
    text_query = f"Chat session_id:{request.session_id}\nConversation history:\n{formatted_history}\nCurrent query: {request.content}"

    # Prepare content based on whether there's an image
    contents = None
    if request.im:
      img = Image.open(BytesIO(open(request.im, "rb").read()))
      contents = [text_query, img]
    else:
      contents = text_query  # Text-only query

    uses_tool = False
    tool_result = None

    # Only try tool detection if we don't have an image
    # Images should be handled directly by DeepSeek
    if not request.im:
      try:
        # Call Gemini API for tool detection only
        api_key = config.get("gemini", "")
        client = genai.Client(api_key=api_key)
        loop = asyncio.get_event_loop()

        # Load tools
        tools_dir = Path("toolLib/tools")
        for tool_file in tools_dir.glob("*.py"):
          if not tool_file.stem.startswith("_"):
            importlib.import_module(f"toolLib.tools.{tool_file.stem}")

        # Configure safety settings
        safety_settings = [
          types.SafetySetting(
            category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
            threshold='BLOCK_NONE',
          )
        ]

        # Tool detection config
        tool_detection_config = types.GenerateContentConfig(
          safety_settings=safety_settings,
          temperature=0.7,
          system_instruction=request.system_prompt[0],
          tools=[types.Tool(function_declarations=ToolRegistry.get_configs())]
        )

        # Call for tool detection
        logger.info(f"Calling Gemini API for tool detection in session {request.session_id}")
        with concurrent.futures.ThreadPoolExecutor() as pool:
          first_response = await loop.run_in_executor(
            pool,
            lambda: client.models.generate_content(
              model='gemini-2.0-flash-exp',
              contents=contents,
              config=tool_detection_config
            )
          )

        # Check if tool is requested - handle the case when no function call is detected
        if first_response and first_response.candidates and len(first_response.candidates) > 0:
          first_part = first_response.candidates[0].content.parts[0]
          uses_tool = (hasattr(first_part, 'function_call') and
                       first_part.function_call is not None and
                       hasattr(first_part.function_call, 'name') and
                       first_part.function_call.name is not None)

          if uses_tool:
            # Extract function details
            function_name = first_part.function_call.name
            function_args = first_part.function_call.args

            # Execute tool
            logger.info(f"Tool usage detected in session {request.session_id}: {function_name}")
            tools = ToolRegistry.get_tools()
            if function_name not in tools:
              raise ValueError(f"Function {function_name} not found in registry")

            tool_class = tools[function_name]
            with concurrent.futures.ThreadPoolExecutor() as pool:
              function_result = await loop.run_in_executor(
                pool,
                lambda: tool_class.execute(**function_args)
              )

            logger.info(f"Tool executed: {function_name}, result: {function_result}")

            # Format tool result for DeepSeek
            tool_result = (
              f"以下是工具 {function_name} 的执行结果，这不是用户输入的一部分，而是自动化查询的返回：\n\n"
              f"<tool>\n{function_result}\n</tool>"
            )
        else:
          logger.info(f"Invalid response from Gemini API for tool detection in session {request.session_id}")
      except Exception as tool_error:
        # If any error occurs during tool detection, log it but continue with DeepSeek
        logger.warning(f"Tool detection error in session {request.session_id}: {str(tool_error)}")
        uses_tool = False

    if not uses_tool:
      logger.info(f"No tool usage detected in session {request.session_id}, proceeding with DeepSeek only")

    # Prepare DeepSeek messages
    # Check if system prompt needs to be embedded
    system_prompt = None
    if not message_service.has_system_prompt_embedded(request.session_id):
      system_prompt = request.system_prompt[1]
      if system_prompt:
        message_service.mark_system_prompt_embedded(request.session_id)

    # Create temporary modified message list for DeepSeek if tool was used
    deepseek_messages = []
    if uses_tool and tool_result:
      # Copy all messages
      for msg in messages:
        if msg == messages[-1] and msg["role"] == "user":
          # For the last user message, append the tool result
          enhanced_content = f"{msg['content']}\n\n{tool_result}"
          deepseek_messages.append({"role": "user", "content": enhanced_content})
        else:
          deepseek_messages.append(msg.copy())
    else:
      deepseek_messages = messages.copy()

    # Format for DeepSeek
    formatted_deepseek_messages = message_service.prepare_deepseek_messages(deepseek_messages, system_prompt)

    if not formatted_deepseek_messages:
      raise ValueError("No valid messages to send to DeepSeek API")

    # Call DeepSeek API for final response
    logger.info(f"Calling DeepSeek API for session {request.session_id}")
    result = await DeepSeekAPI.process_query(
      messages=formatted_deepseek_messages,
      temperature=0.7
    )

    # Calculate processing time
    elapsed_time = time.time() - start_time
    formatted_time = format_duration(elapsed_time)
    logger.info(f"Hybrid request for session {request.session_id} processed in {formatted_time}")

    # Extract results
    reasoning = result.get("reasoning_content", "")
    content = result.get("content", "")

    # Add assistant response to message pool
    message_service.add_message(
      request.session_id,
      "assistant",
      content,
      request.history_limit
    )

    # Return processing result
    return ChatResponse(
      response=content,
      reasoning=reasoning,
      status="success",
      processing_time=formatted_time
    )

  except Exception as e:
    logger.error(f"Error processing hybrid request for session {request.session_id}: {str(e)}")
    # If we got an error and we've added the user message, we should remove it to maintain conversation integrity
    try:
      if lock_acquired:
        messages = message_service.get_messages(request.session_id)
        if messages and messages[-1]["role"] == "user":
          # Remove the last message (which would be the user's) to ensure conversation consistency
          messages = messages[:-1]
          # Overwrite the messages in Redis
          if messages:
            message_service.redis.setex(
              f"chat:{request.session_id}",
              message_service.message_ttl,
              json.dumps(messages)
            )
    except Exception as cleanup_error:
      logger.error(f"Error cleaning up after failed request: {str(cleanup_error)}")

    raise HTTPException(
      status_code=500,
      detail=f"Error processing hybrid request: {str(e)}"
    )
  finally:
    # Ensure lock is released
    if lock_acquired:
      message_service.unlock_session(request.session_id)
      logger.info(f"Session {request.session_id} unlocked after hybrid processing")

@app.post("/clean-session", response_model=CleanSessionResponse)
async def clean_session_endpoint(request: CleanSessionRequest):
  try:
    # Initialize message queue service
    message_service = MessageQueueService(redis_host='localhost', redis_port=6379, redis_db=0)

    # Clear session
    success = message_service.clear_session(request.session_id)
    logger.info(f"Session {request.session_id} cleaned, success: {success}")

    if success:
      return CleanSessionResponse(
        status="success",
        message=f"Session {request.session_id} has been cleared successfully"
      )
    else:
      return CleanSessionResponse(
        status="warning",
        message=f"Session {request.session_id} may not exist or was already cleared"
      )

  except Exception as e:
    logger.error(f"Error cleaning session {request.session_id}: {str(e)}")
    raise HTTPException(
      status_code=500,
      detail=f"Error clearing session: {str(e)}"
    )


@app.get("/health")
async def health_check():
  localtime = time.strftime("%H:%M:%S", time.localtime(time.time()))
  return {"status": f"{localtime} | OK"}


if __name__ == "__main__":
  import uvicorn

  config = load_config()
  port = 5888

  # 使用单进程模式运行
  uvicorn.run(
    app,
    host="0.0.0.0",
    port=port,
    log_level="info"
  )
