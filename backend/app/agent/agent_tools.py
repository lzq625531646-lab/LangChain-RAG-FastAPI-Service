from typing import List, Optional
from contextvars import ContextVar

from langchain_core.tools import tool

from app.core.logger_handler import logger
from app.rag.rag_service import RagService
from app.rag.reorder_service import reorder_service
from app.utils.auth_utils import decode_django_jwt

import datetime

current_user_id_var: ContextVar[str] = ContextVar('current_user_id', default=None)
current_user_token_var: ContextVar[str] = ContextVar('current_user_token', default=None)
thinking_callback_var: ContextVar[Optional[callable]] = ContextVar('thinking_callback', default=None)

def set_current_user_id(user_id: str):
    """设置当前用户ID到上下文"""
    current_user_id_var.set(user_id)

def set_current_user_token(token: str):
    """设置当前用户JWT到上下文"""
    current_user_token_var.set(token)

def get_current_user_id_from_context() -> str:
    """从上下文获取当前用户ID"""
    return current_user_id_var.get()

def get_current_user_token_from_context() -> str:
    """从上下文获取当前用户JWT"""
    return current_user_token_var.get()

def set_thinking_callback(callback):
    """设置思考过程回调到上下文"""
    thinking_callback_var.set(callback)

def get_thinking_callback_from_context():
    """从上下文获取思考过程回调"""
    return thinking_callback_var.get()

@tool(description="用于从向量数据库里检索文档并生成摘要，返回包含文档列表和摘要的结果。返回格式为：'摘要: [摘要内容]\n\n检索到的文档列表:\n1. [文档1内容]\n2. [文档2内容]\n...'。注意：文档已经过自动重排序，无需再调用重排序工具")
async def rag_summary_tools(query: str, user_id: str = None) -> str:
    """RAG 摘要工具"""
    effective_user_id = user_id or get_current_user_id_from_context()
    if not effective_user_id:
        return "错误: 无法确定用户身份，请提供有效的user_id"
    
    thinking_callback = get_thinking_callback_from_context()
    result = await RagService(effective_user_id, thinking_callback=thinking_callback).get_documents_and_summary(query)
    documents = result.get("documents", [])
    summary = result.get("summary", "")

    formatted_result = f"摘要: {summary}\n\n"
    formatted_result += "检索到的文档列表（已重排序）:\n"
    for i, doc in enumerate(documents, 1):
        formatted_result += f"{i}. {doc}\n"

    return formatted_result

@tool(description="用于对文档列表进行重排序，传入查询语句query和文档列表documents，返回重排序后的文档列表，包含文档内容和相似度。注意：rag_summary_tools已内置重排序功能，通常不需要单独调用此工具")
async def reorder_documents_tools(query: str, documents: List[str]) -> str:
    """重排序文档工具"""
    thinking_callback = get_thinking_callback_from_context()
    result = await reorder_service.reorder_documents(query, documents, thinking_callback=thinking_callback)
    if result["success"]:
        # 格式化返回结果
        formatted_result = await reorder_service.format_reorder_result(result["documents"])
        # 记录日志
        logger.info(formatted_result)
        return formatted_result
    else:
        return f"重排序失败: {result['error']}"

@tool(description="当用户明确问自己的ID和用户名时，获取当前登录用户的用户ID和用户名。通常不需要用户额外提供JWT token，优先使用系统上下文中的当前JWT")
async def get_user_info_tools(token: Optional[str] = None) -> str:
    """获取用户信息工具"""
    effective_token = token or get_current_user_token_from_context()
    if not effective_token:
        return "错误: 当前请求上下文中没有JWT token，无法获取用户信息"

    logger.info("开始解析当前用户JWT以获取用户信息")
    payload = decode_django_jwt(effective_token)
    if payload:
        user_id = payload.get("user_id", "未知")
        user_name = payload.get("user_name", "未知")
        return f"用户信息：\n- 用户ID: {user_id}\n- 用户名: {user_name}"
    else:
        return "无法解析JWT token，无法获取用户信息"


@tool(description="用于获取天气信息，需要提供城市名称作为参数，你需要从用户输入中提取城市名称，是str类型")
async def get_weather_tools(city: str = None) -> str:
    """获取天气工具"""
    if not city:
        return "请提供城市名称"
    return f"【{city}】的天气是晴朗的"


@tool(description="用于获取当前年月日时分的工具")
async def what_time_is_now() -> str:
    """获取当前年月日时分的工具"""
    return f"当前时间是：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"


@tool(description="用于获取某个款式的销售件数。需要提供款式编号（style），返回该款式的累计销售件数")
async def get_sales_volume(style: str) -> str:
    """根据款式编号查询销售件数"""
    if not style or not style.strip():
        return "错误：请提供款式编号"

    mock_data = {
        "1": {"name": "经典款", "sales": 1280},
        "2": {"name": "时尚款", "sales": 856},
        "3": {"name": "限定款", "sales": 320},
    }

    result = mock_data.get(style)
    if result:
        return f"款式 {style}（{result['name']}）的累计销售件数为：{result['sales']} 件"
    else:
        return f"未找到款式编号为「{style}」的销售数据，请确认款式编号是否正确"


