from Lib import *
from Lib.core import PluginManager, ConfigManager

logger = Logger.get_logger()

plugin_info = PluginManager.PluginInfo(
  NAME="Listener",
  AUTHOR="MMG",
  VERSION="1.0.0",
  DESCRIPTION="允许机器人框架向外部暴露监听接口",
  HELP_MSG="自动化插件"
)

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
import uvicorn
import platform
import socket
from datetime import datetime
import os

# 创建FastAPI应用
app = FastAPI(
    title="简易API服务",
    description="一个基于FastAPI的简单服务，运行在6222端口",
    version="1.0.0"
)

# 健康检查端点
@app.get("/health", tags=["健康检查"])
async def health_check():
    """
    健康检查端点，返回服务器状态信息
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "hostname": socket.gethostname(),
        "system_info": {
            "os": platform.system(),
            "python_version": platform.python_version(),
        }
    }

# 主页面
@app.get("/", tags=["主页"])
async def root():
    """
    欢迎页面，显示API基本信息
    """
    return {
        "message": "欢迎使用FastAPI服务",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "server_time": datetime.now().isoformat()
    }

# 示例数据端点
@app.get("/SendMessage", tags=["示例"])
async def demo_data():

    return {
        "statue": 3,
        "timestamp": datetime.now().isoformat()
    }

# 状态端点
@app.get("/status", tags=["系统状态"])
async def status():
    """
    返回服务器状态信息
    """
    process_id = os.getpid()
    return {
        "server_status": "running",
        "process_id": process_id,
        "uptime": "N/A",  # 实际应用中可以记录启动时间并计算运行时长
        "memory_usage": "N/A",  # 实际应用中可以添加内存使用监控
        "timestamp": datetime.now().isoformat()
    }
# 错误示例端点
@app.get("/demo/error", tags=["示例"])
async def demo_error():
    """
    演示错误响应的端点
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "示例错误",
            "message": "这是一个示例错误响应",
            "timestamp": datetime.now().isoformat()
        }
    )

import threading

# 创建一个函数来启动服务器
def start_server():
    try:
        logger.info("正在启动FastAPI服务，端口:6222...")
        uvicorn.run(app, host="0.0.0.0", port=6222)
        logger.info("服务器已成功启动")
    except Exception as e:
        logger.info(f"服务器启动失败: {str(e)}")

# 在新线程中启动服务器
server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()
logger.info("FastAPI服务器线程已启动")
