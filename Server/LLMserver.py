import time
from io import BytesIO
from PIL import Image
import importlib
from pathlib import Path
from typing import Optional, Any, List, Dict
import json
import redis
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel
from toolLib.tool_configs import ToolRegistry
from openai import OpenAI


class LLMToolchainAsync:
  def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, redis_db: int = 0):
    # 初始化API客户端
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
    api_key = r.get('llm_api_key')
    self.client = genai.Client(api_key=api_key)

    # 加载工具
    self._load_tools()

  def _load_tools(self):
    """加载工具目录下的所有工具"""
    tools_dir = Path("toolLib/tools")
    for tool_file in tools_dir.glob("*.py"):
      if not tool_file.stem.startswith("_"):
        importlib.import_module(f"toolLib.tools.{tool_file.stem}")

  def _create_request_config(self,
                             system_prompt: str,
                             temperature: float = 0.7,
                             safety_settings: Optional[list] = None) -> types.GenerateContentConfig:
    """创建统一的请求配置"""
    if safety_settings is None:
      safety_settings = [
        types.SafetySetting(
          category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
          threshold='BLOCK_NONE',
        )
      ]

    return types.GenerateContentConfig(
      safety_settings=safety_settings,
      temperature=temperature,
      system_instruction=system_prompt,
      tools=[types.Tool(function_declarations=ToolRegistry.get_configs())]
    )

  async def _make_request(self,
                          contents: list,
                          system_prompt: str,
                          temperature: float = 0.7) -> Any:
    """发送请求到LLM"""
    config = self._create_request_config(
      system_prompt=system_prompt,
      temperature=temperature
    )

    return self.client.models.generate_content(
      model='gemini-2.0-flash-exp',
      contents=contents,
      config=config
    )

  async def process_query_async(self,
                                query: any,
                                sys_prompt1: str,
                                sys_prompt2: str,
                                temperature: float = 0.7) -> str:
    """异步处理查询请求"""
    try:
      if isinstance(query, (list, tuple)):
        # 检查是否包含图片元数据
        for item in query:
          if isinstance(item, Image.Image):
            print("检测到图片输入")
          elif isinstance(item, str):
            print("检测到文本输入")
          else:
            print(f"未知输入类型: {type(item)}")
      else:
        # 如果不是集合，确保是字符串
        if not isinstance(query, str):
          raise ValueError(f"查询必须是字符串或包含文本和图片的列表，当前类型: {type(query)}")
        query = [query]  # 转换为列表格式
      # 第一次调用：获取函数调用请求
      first_response = await self._make_request(
        contents=query,
        system_prompt=sys_prompt1,
        temperature=temperature
      )

      # 检查是否有函数调用
      first_part = first_response.candidates[0].content.parts[0]

      # 如果没有函数调用，直接使用sys_prompt2进行回答
      if not hasattr(first_part, 'function_call'):
        final_response = await self._make_request(
          contents=query,
          system_prompt=sys_prompt2,
          temperature=temperature
        )
        return final_response.text

      # 如果有函数调用，继续原有的处理逻辑
      function_name = first_part.function_call.name
      function_args = first_part.function_call.args

      # 执行工具函数
      tools = ToolRegistry.get_tools()
      if function_name not in tools:
        raise ValueError(f"Function {function_name} not found")

      tool_class = tools[function_name]
      function_response = tool_class.execute(**function_args)

      # 创建函数响应
      function_response_part = types.Part.from_function_response(
        name=function_name,
        response={'result': function_response}
      )

      # 最终调用：结合上下文获取完整响应
      final_response = await self._make_request(
        contents=[
          types.Part.from_text(query),
          first_part,
          function_response_part,
        ],
        system_prompt=sys_prompt2,
        temperature=temperature
      )

      return final_response.text

    except Exception as e:
      # 如果出现异常，尝试直接使用sys_prompt2进行回答
      try:
        config = self._create_request_config(
          system_prompt=sys_prompt2,
          temperature=temperature
        )

        response = self.client.models.generate_content(
          model='gemini-2.0-flash-exp',
          contents=[query],
          config=config
        )
        return response.text
      except Exception as inner_e:
        # 如果备用方案也失败，则抛出原始异常
        raise Exception(f"Error processing query: {str(e)}")


async def process_llm_query(
  query: Any,
  sys_prompt1: str,
  sys_prompt2: str,
  temperature: float = 0.7,
  redis_host: str = 'localhost',
  redis_port: int = 6379,
  redis_db: int = 0
) -> str:
  """
  独立的异步函数，用于处理LLM查询

  Args:
      query: 用户输入的查询
      sys_prompt1: 第一次调用时使用的系统提示
      sys_prompt2: 第二次调用时使用的系统提示
      temperature: 随机参数
      redis_host: Redis主机地址
      redis_port: Redis端口
      redis_db: Redis数据库编号

  Returns:
      str: LLM的响应文本
  """
  toolchain = LLMToolchainAsync(redis_host, redis_port, redis_db)
  return await toolchain.process_query_async(
    query=query,
    sys_prompt1=sys_prompt1,
    sys_prompt2=sys_prompt2,
    temperature=temperature
  )


class DeepSeekClient:
  def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, redis_db: int = 0):
    # 从Redis获取API密钥
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
    api_key = "sk-83ddd962dc084190b3f5ef0cb12252a9"

    if not api_key:
      raise ValueError("DeepSeek API key not found in Redis")

    self.client = OpenAI(
      api_key=api_key,
      base_url="https://api.deepseek.com"
    )

  async def process_query(self,
                          messages: List[Dict[str, str]],
                          temperature: float = 0.7) -> Dict[str, str]:
    """
    处理DeepSeek查询

    Args:
        messages: 消息历史列表，格式为[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        temperature: 温度参数

    Returns:
        Dict: 包含reasoning_content和content的字典
    """
    try:
      # 准备请求参数
      params = {
        "model": "deepseek-reasoner",
        "messages": messages,
        "temperature": temperature
      }

      # 发送请求
      response = self.client.chat.completions.create(**params)

      # 提取结果
      reasoning_content = response.choices[0].message.reasoning_content
      content = response.choices[0].message.content

      return {
        "reasoning_content": reasoning_content,
        "content": content
      }

    except Exception as e:
      raise Exception(f"DeepSeek API请求失败: {str(e)}")


# FastAPI应用程序定义
app = FastAPI()


# 请求模型定义
class ChatRequest(BaseModel):
  session_id: str
  content: str
  system_prompt: List[str]
  im: Optional[str] = None
  history_limit: int = 10  # 默认保留最近10条消息


class ChatResponse(BaseModel):
  response: str
  reasoning: Optional[str] = None

class CleanSessionRequest(BaseModel):
  session_id: str


class CleanSessionResponse(BaseModel):
  status: str
  message: str


class MessagePool:
  def __init__(self, redis_client: redis.Redis):
    self.redis = redis_client
    self.message_ttl = 24 * 60 * 60  # 消息保存24小时

  def get_messages(self, session_id: str) -> List[Dict[str, str]]:
    messages_str = self.redis.get(f"chat:{session_id}")
    if messages_str:
      return json.loads(messages_str)
    return []

  def add_message(self, session_id: str, role: str, content: str, history_limit: int):
    messages = self.get_messages(session_id)
    messages.append({"role": role, "content": content})

    # 保留最新的n条消息
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
    """清除特定会话的所有消息和相关标记"""
    # 删除聊天消息
    chat_key = f"chat:{session_id}"
    # 删除系统提示嵌入标记
    system_prompt_key = f"chat:{session_id}:has_system_prompt"

    # 使用pipeline批量删除键
    pipe = self.redis.pipeline()
    pipe.delete(chat_key)
    pipe.delete(system_prompt_key)
    results = pipe.execute()

    # 如果至少有一个键被删除，则认为清理成功
    return any(result == 1 for result in results)

  def prepare_deepseek_messages(self, messages: List[Dict[str, str]], system_prompt: str = None) -> List[
    Dict[str, str]]:
    """
    格式化消息以适合DeepSeek API要求，确保用户和助手消息严格交替
    如果是首次对话，在第一条用户消息中嵌入系统提示
    """
    if not messages:
      return []

    # 标准化所有消息中的角色名
    normalized_messages = []
    for msg in messages:
      role = msg["role"].lower()
      if role in ["user", "assistant"]:
        normalized_messages.append({"role": role, "content": msg["content"]})

    # 如果没有消息，返回空列表
    if not normalized_messages:
      return []

    # 确保消息序列以用户消息开始
    if normalized_messages[0]["role"] != "user":
      normalized_messages = normalized_messages[1:]
      if not normalized_messages:
        return []

    # 创建严格交替的消息序列
    formatted_messages = []
    current_role = "user"

    for i, msg in enumerate(normalized_messages):
      if msg["role"] == current_role:
        # 如果是第一条用户消息且有系统提示，嵌入系统提示
        if i == 0 and current_role == "user" and system_prompt:
          formatted_messages.append({
            "role": "user",
            "content": f"{system_prompt}\n\n{msg['content']}"
          })
        else:
          formatted_messages.append(msg)

        # 切换期望的下一个角色
        current_role = "assistant" if current_role == "user" else "user"
      else:
        # 如果角色不符合预期顺序，跳过此消息直到找到正确角色的消息
        continue

    # 确保消息列表以用户消息结束(如果最后一条是assistant，则移除)
    if formatted_messages and formatted_messages[-1]["role"] == "assistant":
      formatted_messages = formatted_messages[:-1]

    return formatted_messages

  def has_system_prompt_embedded(self, session_id: str) -> bool:
    """检查是否已经在对话中嵌入了系统提示"""
    key = f"chat:{session_id}:has_system_prompt"
    return bool(self.redis.get(key))

  def mark_system_prompt_embedded(self, session_id: str):
    """标记此会话已嵌入系统提示"""
    self.redis.setex(
      f"chat:{session_id}:has_system_prompt",
      self.message_ttl,
      "1"
    )


@app.post("/gemini/chat", response_model=ChatResponse)
async def gemini_chat_endpoint(request: ChatRequest):
  try:
    if len(request.system_prompt) != 2:
      raise HTTPException(
        status_code=400,
        detail="system_prompt must contain exactly 2 elements"
      )

    # 初始化消息池
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    message_pool = MessagePool(redis_client)

    # 添加用户消息到消息池
    message_pool.add_message(
      request.session_id,
      "user",
      request.content,
      request.history_limit
    )

    # 获取历史消息并格式化
    messages = message_pool.get_messages(request.session_id)
    formatted_history = message_pool.format_messages(messages)

    # 将格式化的历史消息作为context传递给LLM
    query = f"Chat session_id:{request.session_id}\nConversation history:\n{formatted_history}\nCurrent query: {request.content}"
    if request.im:
      print("code image")
      print(request.im)
      img = Image.open(BytesIO(open(request.im, "rb").read()))
      # img.show() 纠错检查点
      query = [query, img]
    else:
      query = [query]

    result = await process_llm_query(
      query=query,
      sys_prompt1=request.system_prompt[0],
      sys_prompt2=request.system_prompt[1]
    )

    # 添加助手响应到消息池
    message_pool.add_message(
      request.session_id,
      "assistant",
      result,
      request.history_limit
    )

    return ChatResponse(response=result)

  except Exception as e:
    raise HTTPException(
      status_code=500,
      detail=f"Error processing request: {str(e)}"
    )


@app.post("/deepseek/chat", response_model=ChatResponse)
async def deepseek_chat_endpoint(request: ChatRequest):
  try:
    # 初始化消息池和DeepSeek客户端
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    message_pool = MessagePool(redis_client)
    deepseek_client = DeepSeekClient(redis_host='localhost', redis_port=6379, redis_db=0)

    # 添加用户消息到消息池
    message_pool.add_message(
      request.session_id,
      "user",
      request.content,
      request.history_limit
    )

    # 获取历史消息
    raw_messages = message_pool.get_messages(request.session_id)

    # 检查是否需要嵌入系统提示（仅对第一次对话）
    system_prompt = None
    if not message_pool.has_system_prompt_embedded(request.session_id):
      system_prompt = request.system_prompt[1] if len(request.system_prompt) > 1 else None
      # 标记已嵌入系统提示
      if system_prompt:
        message_pool.mark_system_prompt_embedded(request.session_id)

    # 准备符合DeepSeek要求的消息格式（确保严格的用户-助手交替顺序）
    deepseek_messages = message_pool.prepare_deepseek_messages(raw_messages, system_prompt)

    # 确保当前有要发送的消息
    if not deepseek_messages:
      raise ValueError("No valid messages to send to DeepSeek API")

    # 调用DeepSeek API
    result = await deepseek_client.process_query(
      messages=deepseek_messages,
      temperature=0.7
    )

    # 提取结果
    reasoning = result.get("reasoning_content", "")
    content = result.get("content", "")

    # 添加助手响应到消息池
    message_pool.add_message(
      request.session_id,
      "assistant",
      content,
      request.history_limit
    )

    return ChatResponse(response=content, reasoning=reasoning)

  except Exception as e:
    raise HTTPException(
      status_code=500,
      detail=f"Error processing DeepSeek request: {str(e)}"
    )


@app.post("/clean-session", response_model=CleanSessionResponse)
async def clean_session_endpoint(request: CleanSessionRequest):
  """
  清除特定会话的所有聊天记录

  Args:
      request: 包含se ssion_id的请求对象

  Returns:
      CleanSessionResponse: 操作状态响应
  """
  try:
    # 初始化消息池
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    message_pool = MessagePool(redis_client)

    # 清除会话
    success = message_pool.clear_session(request.session_id)

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
    raise HTTPException(
      status_code=500,
      detail=f"Error clearing session: {str(e)}"
    )


@app.get("/health")
async def health_check():
  localtime = time.strftime("%H：%M：%S", time.localtime(time.time()))
  return {"status": f"{localtime} | OK"}




if __name__ == "__main__":
  import uvicorn

  port = 5888
  uvicorn.run(
    app,
    host="0.0.0.0",
    port=port,
    log_level="info"  # 添加日志级别
  )
