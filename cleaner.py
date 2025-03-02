import os
import sys
import subprocess
import time
import signal
import platform


def get_pids_by_port(port):
  """获取占用指定端口的进程ID列表"""
  pids = []
  system = platform.system()

  try:
    if system == "Windows":
      # Windows 系统使用 netstat 命令
      output = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode('utf-8')
      for line in output.strip().split('\n'):
        if f":{port}" in line:
          parts = line.strip().split()
          if len(parts) > 4:
            pid = parts[4]
            if pid not in pids and pid != str(os.getpid()):
              pids.append(pid)
    else:
      # Linux/macOS 系统使用 lsof 命令
      output = subprocess.check_output(f'lsof -i :{port}', shell=True).decode('utf-8')
      for line in output.strip().split('\n')[1:]:  # 跳过标题行
        parts = line.strip().split()
        if len(parts) > 1:
          pid = parts[1]
          if pid not in pids and pid != str(os.getpid()):
            pids.append(pid)
  except subprocess.CalledProcessError:
    # 如果命令执行失败，可能是没有进程使用该端口
    pass

  return pids


def kill_process(pid):
  """杀死指定进程ID的进程"""
  system = platform.system()
  try:
    if system == "Windows":
      subprocess.run(f'taskkill /F /PID {pid}', shell=True, check=True)
      print(f"已终止进程: PID {pid}")
    else:
      subprocess.run(f'kill -9 {pid}', shell=True, check=True)
      print(f"已终止进程: PID {pid}")
    return True
  except subprocess.CalledProcessError as e:
    print(f"终止进程 {pid} 时出错: {e}")
    return False


def main():
  """主函数，清理指定端口的进程"""
  ports = [5705, 5706]
  print(f"开始清理端口 {', '.join(map(str, ports))} 的进程...")

  for port in ports:
    pids = get_pids_by_port(port)
    if pids:
      print(f"端口 {port} 被以下进程占用: {', '.join(pids)}")
      for pid in pids:
        kill_process(pid)
    else:
      print(f"端口 {port} 没有被任何进程占用")

  print("清理完成，脚本将在3秒后结束...")
  time.sleep(3)

  # 释放自己的进程
  print(f"正在结束当前进程 (PID: {os.getpid()})...")

  # 优雅地结束当前进程
  os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
  main()
