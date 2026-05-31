from typing import Callable, TypeVar, Generic
from functools import wraps

from app.db.redis_config import get_redis_cache_json, get_redis_cache_str, set_redis_cache, redis_client
from app.core.logger_handler import get_logger

T = TypeVar('T')
logger = get_logger(__name__)


class RedisCache(Generic[T]):
    """
    Redis缓存管理类

    提供通用的缓存操作: 自动缓存和过期时间设置
    """

    @staticmethod
    async def get_or_set(
            key: str,
            func: Callable[..., T],
            *args,
            expire: int = 3600,
            **kwargs
    ) -> T:
        """
        获取缓存，如果缓存不存在则执行函数并缓存结果

        :param key: 缓存键
        :param func: 要执行的函数
        :param args: 函数参数
        :param kwargs: 函数关键字参数
        :param expire: 缓存过期时间(秒)
        :return: 函数执行结果
        """
        # 尝试从缓存获取
        # 无论key是什么类型，都统一转换为字符串
        cache_key = str(key)
        # 先尝试以JSON格式获取
        cached_data = await get_redis_cache_json(cache_key)
        # print(f"【RedisCache】尝试以JSON格式获取缓存，key: {cache_key}")
        # 如果JSON解析失败，尝试以字符串格式获取
        if cached_data is None:
            cached_data = await get_redis_cache_str(cache_key)
            # print(f"【RedisCache】以字符串格式获取缓存，key: {cache_key}")

        if cached_data is not None:
            logger.debug("Redis缓存命中 key=%s value_type=%s", cache_key, type(cached_data).__name__)
            return cached_data

        logger.debug("Redis缓存未命中 key=%s", cache_key)

        # 缓存不存在，执行函数
        result = await func(*args, **kwargs)

        # 将结果转换为可序列化的格式
        def convert_to_serializable(obj):
            """将对象转换为可序列化的格式"""
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif hasattr(obj, '__dict__'):
                # 处理模型对象
                obj_dict = {}
                for key, value in obj.__dict__.items():
                    # 排除内部属性和不可序列化的属性
                    if not key.startswith('_') and not hasattr(value, '__dict__'):
                        obj_dict[key] = convert_to_serializable(value)
                return obj_dict
            else:
                # 其他类型尝试转换为字符串
                try:
                    return str(obj)
                except Exception as e:
                    logger.exception("转换对象为字符串时出错: %s", e)
                    return None

        # 转换结果
        serializable_result = convert_to_serializable(result)

        # 缓存结果
        logger.debug("设置Redis缓存 key=%s value_type=%s expire=%s", cache_key, type(serializable_result).__name__, expire)
        success = await set_redis_cache(cache_key, serializable_result, expire)
        logger.debug("Redis缓存设置结果 key=%s success=%s", cache_key, success)
        return result

    @staticmethod
    def cache_key(prefix: str, *args, **kwargs) -> str:
        """
        生成缓存键

        :param prefix: 缓存键前缀
        :param args: 函数参数
        :param kwargs: 函数关键字参数
        :return: 生成的缓存键
        """
        parts = [prefix]

        # 添加位置参数, 排除数据库会话
        for arg in args:
            if arg is not None and not hasattr(arg, 'execute'):
                parts.append(str(arg))

        # 添加关键字参数
        for key, value in sorted(kwargs.items()):
            if value is not None and key != 'db':
                parts.append(f"{key}:{value}")

        return ":".join(parts)

    @staticmethod
    async def delete(key: str) -> bool:
        """
        删除缓存

        :param key: 缓存键
        :return: 是否删除成功
        """
        try:
            await redis_client.delete(key)
            logger.debug("删除Redis缓存 key=%s success=True", key)
            return True
        except Exception as e:
            logger.exception("删除redis缓存失败 key=%s error=%s", key, e)
            return False

    @staticmethod
    async def delete_pattern(pattern: str) -> int:
        """
        根据模式删除缓存

        :param pattern: 缓存键模式，支持通配符
        :return: 删除的缓存数量
        """
        try:
            keys = await redis_client.keys(pattern)
            if keys:
                deleted = await redis_client.delete(*keys)
                logger.debug("按模式删除Redis缓存 pattern=%s matched=%s deleted=%s", pattern, len(keys), deleted)
                return deleted
            logger.debug("按模式删除Redis缓存 pattern=%s matched=0", pattern)
            return 0
        except Exception as e:
            logger.exception("删除redis缓存失败 pattern=%s error=%s", pattern, e)
            return 0


def cache_with_redis(prefix: str, expire: int = 3600):
    """
    Redis缓存装饰器

    :param prefix: 缓存键前缀
    :param expire: 缓存过期时间(秒)
    :return: 装饰器函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            key = RedisCache.cache_key(prefix, *args, **kwargs)

            # 使用RedisCache获取或设置缓存
            return await RedisCache.get_or_set(key, func, *args, expire=expire, **kwargs)
        return wrapper
    return decorator
