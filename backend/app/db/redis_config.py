import json
import os
from typing import Any

import redis.asyncio as redis

from app.core.logger_handler import get_logger

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "3"))
logger = get_logger(__name__)

# 全局redis客户端对象
redis_client = None

async def connect_redis():
    """连接Redis"""
    global redis_client
    if redis_client is None:
        logger.info("正在创建Redis连接，host=%s port=%s db=%s", REDIS_HOST, REDIS_PORT, REDIS_DB)
        redis_client = redis.Redis(
            host=REDIS_HOST, # redis主机地址
            port=REDIS_PORT, # redis端口号
            db=REDIS_DB,     # redis数据库编号(0-15)
            decode_responses=True # 是否对返回值进行解码(True:返回字符串,False:返回字节)
        )
    return redis_client

async def close_redis():
    """关闭Redis连接"""
    global redis_client
    if redis_client:
        await redis_client.aclose()
        redis_client = None
        logger.info("Redis连接已关闭")

async def check_redis_connection() -> bool:
    """检查Redis连接"""
    try:
        redis_client = await connect_redis()
        await redis_client.ping()
        logger.info("Redis连接检查成功")
        return True
    except Exception as e:
        logger.exception("Redis连接失败: %s", e)
        return False

# 设置和读取redis
async def get_redis_cache_str(key: str) -> str | None:
    """根据key获取redis缓存 (字符串类型)"""
    try:
        redis_client = await connect_redis()
        result = await redis_client.get(key)
        logger.debug("读取Redis字符串缓存 key=%s hit=%s", key, result is not None)
        return result
    except Exception as e:
        logger.exception("获取redis缓存失败 key=%s error=%s", key, e)
        return None

async def get_redis_cache_json(key: str) -> dict | None:
    """根据key获取redis缓存 (字典或列表类型)"""
    try:
        redis_client = await connect_redis()
        data = await redis_client.get(key)
        if data:
            logger.debug("读取Redis JSON缓存 key=%s hit=True", key)
            return json.loads(data)
        logger.debug("读取Redis JSON缓存 key=%s hit=False", key)
        return None
    except Exception as e:
        logger.exception("获取redis的JSON缓存失败 key=%s error=%s", key, e)
        return None

async def set_redis_cache(key: str, value: Any, expire: int = 3600) -> bool:
    """
    根据key设置redis缓存

    :param key: 缓存键
    :param value: 缓存值
    :param expire: 过期时间(秒)
    :return: None
    """
    try:
        redis_client = await connect_redis()
        if isinstance(value, str):
            # 如果是字符串，直接设置缓存
            await redis_client.set(key, value, ex=expire)
        elif isinstance(value, (dict, list)):
            # 如果是字典或列表，转为json字符串在设置缓存
            await redis_client.set(key, json.dumps(value, ensure_ascii=False), ex=expire)
        else:
            # 其他类型，尝试转换为字符串
            await redis_client.set(key, str(value), ex=expire)
        logger.debug("设置Redis缓存成功 key=%s expire=%s value_type=%s", key, expire, type(value).__name__)
        return True

    except Exception as e:
        logger.exception("设置redis缓存失败 key=%s error=%s", key, e)
        return False
