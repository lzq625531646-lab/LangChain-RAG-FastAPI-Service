import os
import json
import asyncio
from langsmith import traceable
from typing import Any, List, Optional, AsyncGenerator

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

try:
    from langchain_dashscope import ChatTongyi
except Exception:
    from langchain_community.chat_models import ChatTongyi

from app.agent.agent_middleware import get_middleware
from app.agent.agent_tools import rag_summary_tools, get_weather_tools, what_time_is_now, get_user_info_tools, \
    reorder_documents_tools, set_current_user_id, set_current_user_token, set_thinking_callback, get_sales_volume
from app.core.logger_handler import logger
from app.core.request_context import set_user_context
from app.services import session_manager as sm
from app.utils.prompt_loader import load_prompt


class AgentFactory:
    """
    生产 Agent 工厂类
    支持：
    - 每次调用创建全新的 LangChain v1 Agent Runnable
    - 动态注入工具、提示词、模型配置
    - 支持异步流式调用
    """

    def __init__(
            self,
            model: str = "qwen3-max",
            api_key: Optional[str] = None,
            default_tools: Optional[List[BaseTool]] = None,
            default_middleware: Optional[List] = None,
            default_system_prompt: Optional[str] = None,
    ):
        """
        初始化工厂配置（仅配置，不创建实例）
        :param model: 默认模型名称
        :param api_key: 默认 API Key（不传则从env读取）
        :param default_tools: 默认工具列表
        :param default_system_prompt: 默认系统提示词
        """
        self.model = model
        self.api_key = api_key or os.getenv("CHAT_API_KEY")
        self.default_tools = default_tools or self._get_default_tools()
        self.default_middleware = default_middleware or self._get_default_middleware()
        self.default_system_prompt = default_system_prompt or self._get_default_system_prompt()

    @staticmethod
    def _get_default_tools() -> List[BaseTool]:
        """获取默认工具列表"""
        return [
            rag_summary_tools,
            get_weather_tools,
            what_time_is_now,
            get_user_info_tools,
            reorder_documents_tools,
            get_sales_volume
        ]

    def _get_default_middleware(self) -> List:
        """获取默认中间件列表"""
        return get_middleware()

    @staticmethod
    def _get_default_system_prompt() -> str:
        """获取默认系统提示词"""
        return load_prompt('main_prompt')

    def _create_chat_model(self, custom_model: Optional[str] = None):
        """内部方法：根据LLM_TYPE创建聊天模型实例"""
        llm_type = os.getenv("LLM_TYPE", "ALIYUN").upper()
        
        if llm_type == "OLLAMA":
            model_name = custom_model or os.getenv("OLLAMA_MODEL_NAME", self.model)
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            logger.info(f"🤖 Agent使用Ollama模型: {model_name}")
            
            return ChatOllama(
                model=model_name,
                base_url=base_url,
                streaming=True,
                top_p=0.7,
            )
        
        elif llm_type == "ALIYUN":
            api_key = os.getenv("ALIYUN_ACCESS_KEY_SECRET")
            base_url = os.getenv("ALIYUN_BASE_URL")
            model_name = custom_model or os.getenv("ALIYUN_MODEL_NAME", self.model)
            
            logger.info(f"🤖 Agent使用阿里云百炼模型: {model_name}")
            
            return ChatTongyi(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                streaming=True,
                top_p=0.7,
            )
        
        else:
            raise ValueError(f"不支持的LLM_TYPE: {llm_type}，可选值: ALIYUN, OLLAMA")

    def create_agent_executor(
            self,
            custom_tools: Optional[List[BaseTool]] = None,
            custom_model: Optional[str] = None,
            custom_system_prompt: Optional[str] = None,
            verbose: bool = True,
            return_intermediate_steps: bool = True,
            **kwargs
    ) -> Runnable:
        """
        核心工厂方法：创建全新的 LangChain v1 Agent 实例
        每次调用都会生成新的实例，彻底避免全局状态污染

        :param custom_tools: 自定义工具列表（覆盖默认）
        :param custom_model: 自定义模型（覆盖默认）
        :param custom_system_prompt: 自定义系统提示词（覆盖默认）
        :param verbose: 保留参数，兼容旧调用方
        :param return_intermediate_steps: 保留参数，兼容旧调用方
        :param kwargs: 其他 create_agent 参数
        :return: 全新的 Agent Runnable 实例
        """
        chat_model = self._create_chat_model(custom_model)
        tools = custom_tools or self.default_tools
        system_prompt = custom_system_prompt or self.default_system_prompt

        return create_agent(
            model=chat_model,
            tools=tools,
            system_prompt=system_prompt,
            middleware=self.default_middleware,
            **kwargs
        )


# 初始化全局工厂配置
agent_factory = AgentFactory()


def get_agent_executor():
    """
    获取Agent Runnable实例，用于LangGraph
    :return: Agent Runnable实例
    """
    return agent_factory.create_agent_executor()


def _build_messages(query: str, history: Optional[List[tuple]] = None) -> List[BaseMessage]:
    """将项目里的二元组历史转换成 LangChain v1 messages 输入。"""
    messages: List[BaseMessage] = []
    if history:
        for user_msg, assistant_msg in history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=assistant_msg))
    messages.append(HumanMessage(content=query))
    return messages


def _extract_response(agent_output: dict[str, Any]) -> str:
    messages = agent_output.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return "抱歉，我无法理解您的请求。"


def _extract_steps(agent_output: dict[str, Any]) -> List[dict[str, Any]]:
    steps = []
    for message in agent_output.get("messages", []):
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                steps.append({
                    "thought": "",
                    "tool": tool_call.get("name"),
                    "tool_input": tool_call.get("args", {}),
                    "tool_output": ""
                })
        elif isinstance(message, ToolMessage):
            for step in reversed(steps):
                if step.get("tool_output") == "":
                    step["tool_output"] = str(message.content)
                    break
    return steps


def _log_tool_steps(prefix: str, steps: List[dict[str, Any]]) -> None:
    for step in steps:
        logger.info(f"\n\n{prefix} [调用工具] {step.get('tool')}")
        logger.info(f"{prefix} [工具输入] {step.get('tool_input')}")
        logger.info(f"{prefix} [工具结果] {step.get('tool_output')}\n")


async def get_agent_response(
        query: str,
        history: Optional[List[tuple]] = None,
        user_id: Optional[str] = None,
        custom_tools: Optional[List[BaseTool]] = None,
        token: Optional[str] = None,
        **kwargs
):
    """
    获取 Agent 响应（使用工厂创建实例）
    :param query: 用户查询
    :param history: 会话历史 [(user_msg, assistant_msg), ...]
    :param user_id: 用户ID
    :param custom_tools: 自定义工具（可选，用于动态切换工具）
    :param kwargs: 其他工厂参数
    :return: 响应结果
    """
    if user_id:
        set_current_user_id(user_id)
        set_user_context(user_id=user_id)
    if token:
        set_current_user_token(token)

    try:
        # 1. 从工厂获取全新的 Executor 实例
        agent_executor = agent_factory.create_agent_executor(custom_tools=custom_tools, **kwargs)

        agent_output = await agent_executor.ainvoke({
            "messages": _build_messages(query, history)
        })
        steps = _extract_steps(agent_output)
        if not steps:
            logger.info("【Agent响应】本轮未调用任何工具")
        else:
            _log_tool_steps("【Agent响应】", steps)

        return {
            "response": _extract_response(agent_output),
            "steps": steps
        }

    except Exception as e:
        logger.error(f"Agent 执行错误: {str(e)}", exc_info=True)
        return {
            "response": f"抱歉，处理您的请求时出现了错误: {str(e)}",
            "steps": []
        }

@traceable
async def get_agent_stream_response(
        query: str,
        session_id: str,
        user_id: str,
        custom_tools: Optional[List[BaseTool]] = None,
        token: Optional[str] = None,
        **kwargs
) -> AsyncGenerator[str, None]:
    """
    获取 Agent 流式响应（包含思考过程，实时推送）
    :param query: 用户查询
    :param session_id: 会话 ID
    :param user_id: 用户 ID
    :param custom_tools: 自定义工具（可选）
    :param kwargs: 其他参数
    :return: 流式响应生成器
    """
    
    thinking_queue = asyncio.Queue()
    agent_result_holder = {"response": None, "error": None}
    agent_done = asyncio.Event()
    set_user_context(user_id=user_id, session_id=session_id)
    
    async def thinking_callback(data: dict):
        """思考过程回调函数，将事件放入队列"""
        logger.info(f"【思考过程】{data.get('stage', 'unknown')}: {data.get('content', '')}")
        await thinking_queue.put(data)
    
    async def run_agent():
        """在独立任务中执行 Agent"""
        try:
            set_current_user_id(user_id)
            if token:
                set_current_user_token(token)
            set_thinking_callback(thinking_callback)
            
            history = await sm.session_manager.get_history(session_id, user_id)
            logger.info(f"【Agent流式响应】获取会话历史成功，历史记录数: {len(history)}")
            
            agent_executor = agent_factory.create_agent_executor(custom_tools=custom_tools, **kwargs)

            agent_output = await agent_executor.ainvoke({
                "messages": _build_messages(query, history)
            })
            steps = _extract_steps(agent_output)
            if not steps:
                logger.info("【Agent流式响应】本轮未调用任何工具")
            else:
                _log_tool_steps("【Agent流式响应】", steps)

            agent_result_holder["response"] = _extract_response(agent_output)
        except Exception as e:
            logger.error(f"【Agent流式响应】Agent执行失败: {e}", exc_info=True)
            agent_result_holder["error"] = str(e)
        finally:
            agent_done.set()
    
    # 启动 Agent 执行任务
    agent_task = asyncio.create_task(run_agent())
    
    try:
        logger.info(f"【Agent流式响应】开始处理请求，用户ID: {user_id}, 会话ID: {session_id}, 查询: {query}")

        # 先发送初始响应
        yield f"data: {json.dumps({'type': 'response', 'content': '', 'session_id': session_id}, ensure_ascii=False)}\n\n"
        
        # 持续监听队列并实时推送思考事件，同时等待 Agent 完成
        while not agent_done.is_set():
            try:
                # 使用短超时轮询队列，实现实时推送
                event = await asyncio.wait_for(thinking_queue.get(), timeout=0.1)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                thinking_queue.task_done()
            except asyncio.TimeoutError:
                # 超时是正常的，继续等待
                continue
        
        # Agent 已完成，推送队列中剩余的所有思考事件
        while not thinking_queue.empty():
            try:
                event = thinking_queue.get_nowait()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                thinking_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        # 等待 agent_task 完全结束
        await agent_task
        
        if agent_result_holder["error"]:
            error_message = f"错误: {agent_result_holder['error']}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_message, 'session_id': session_id}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            return
        
        response = agent_result_holder["response"]
        
        # 添加到会话历史
        await sm.session_manager.add_message(session_id, user_id, query, response)
        logger.info(f"【Agent流式响应】添加到会话历史成功")
        
        # 发送回答内容
        for char in response:
            yield f"data: {json.dumps({'type': 'response', 'content': char}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)
        
        # 发送结束标记
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id}, ensure_ascii=False)}\n\n"
        logger.info(f"【Agent流式响应】处理完成，会话ID: {session_id}")
        
    except Exception as e:
        logger.error(f"【Agent流式响应】处理请求失败: {e}", exc_info=True)
        
        # 取消 agent 任务
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        
        error_message = f"错误: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'content': error_message, 'session_id': session_id}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
