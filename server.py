import threading
import time

from fastapi import FastAPI, WebSocket, Query, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import asyncio

import argparse
import json
import logging
import socket
from typing import Dict, Tuple


# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler('tmp/bailing.log')  # 文件输出
    ]
)
from bailing import robot

# 获取根 logger
logger = logging.getLogger(__name__)



parser = argparse.ArgumentParser(description="Description of your script.")

# Add arguments
parser.add_argument('--config_path', type=str, help="配置文件", default="config/config.yaml")

# Parse arguments
args = parser.parse_args()
config_path = args.config_path


app = FastAPI()
TIMEOUT = 600  # 60 秒不活跃断开
active_robots: Dict[str, list] = {}

async def cleanup_task():
    while True:
        now = time.time()
        for uid, (robot_instance, ts) in list(active_robots.items()):
            if now - ts > TIMEOUT:
                try:
                    robot_instance.recorder.stop_recording()
                    robot_instance.shutdown()
                    logger.info(f"{uid} 对应的robot已释放")
                except Exception as e:
                    logger.info(f"{uid} 对应的robot释放 出错: {e}")
                active_robots.pop(uid, None)
        await asyncio.sleep(10)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cleanup_task())
    yield
    task.cancel()
    await task

app = FastAPI(lifespan=lifespan)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str = Query(...)):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    logger.info("WebSocket连接已建立")
    if user_id not in active_robots:
        active_robots[user_id] = [robot.Robot(config_path, websocket, loop), time.time()]
        threading.Thread(target=active_robots[user_id][0].run, daemon=True).start()
        #active_robots[user_id][0].run()
    robot_instance = active_robots[user_id][0]

    try:
        # 模拟处理流程
        while True:
            msg = await websocket.receive()

            if "bytes" in msg:
                robot_instance.recorder.put_audio(msg["bytes"])
            elif "text" in msg:
                logger.info(f"收到请求{msg}")
                msg_js = json.loads(msg["text"])
                if msg_js["type"] == "playback_status":
                    # 播放中
                    if msg_js["status"]== "playing" or msg_js["queue_size"]>0:
                        logger.info(f"[Client] status: {msg}")
                        robot_instance.player.set_playing_status(True)
                    else: # 未播放
                        robot_instance.player.set_playing_status(False)
                else:
                    logger.warning(f"未知指令：{msg}")
            active_robots[user_id][1] = time.time()

    except WebSocketDisconnect:
        logger.info("客户端断开连接")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        # 清理资源
        #robot_instance.recorder.stop_recording()
        #robot_instance.shutdown()
        logger.info("WebSocket连接已关闭")

# 托管前端静态文件
app.mount("/", StaticFiles(directory="static", html=True), name="static")

def get_lan_ip():
    try:
        # 创建一个UDP套接字
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # 连接到Google DNS
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return "无法获取IP: " + str(e)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001, help="Port to run the server on")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    lan_ip = get_lan_ip()
    print(f"\n请在局域网中使用以下地址访问:")
    print(f"https://{lan_ip}:8034\n")
    # 生成自签名证书 (开发环境)
    # openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ws_ping_interval=20,
        ws_ping_timeout=100
    )