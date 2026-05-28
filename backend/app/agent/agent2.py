import asyncio
import os
from typing import List, Optional, AsyncGenerator

from django.contrib.gis.db.backends.postgis.pgraster import chunk
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.schema.runnable import history
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from app.agent.agent_middleware import get_middleware
from app.agent.agent_tools import rag_summary_tools, get_weather_tools, what_time_is_now, get_user_info_tools, \
    reorder_documents_tools, set_current_user_id, set_thinking_callback
from app.core.logger_handler import logger
from app.services import session_manager
from app.utils.prompt_loader import load_prompt


class AgentFactory:
    """
    生产 Agent 工厂类
    支持：
    - 每次调用创建全新的 AgentExecutor 实例
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

    def _create_prompt(self, custom_system_prompt: Optional[str] = None) -> ChatPromptTemplate:
        """内部方法：创建提示词模板"""
        return ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    def create_agent_executor(
            self,
            custom_tools: Optional[List[BaseTool]] = None,
            custom_model: Optional[str] = None,
            custom_system_prompt: Optional[str] = None,
            verbose: bool = True,
            return_intermediate_steps: bool = True,
            **kwargs
    ) -> AgentExecutor:
        """
        核心工厂方法：创建全新的 AgentExecutor 实例
        每次调用都会生成新的实例，彻底避免全局状态污染

        :param custom_tools: 自定义工具列表（覆盖默认）
        :param custom_model: 自定义模型（覆盖默认）
        :param custom_system_prompt: 自定义系统提示词（覆盖默认）
        :param verbose: 是否打印详细日志
        :param return_intermediate_steps: 是否返回中间步骤
        :param kwargs: 其他 AgentExecutor 参数
        :return: 全新的 AgentExecutor 实例
        """
        # 1. 创建组件（每次都重新创建，避免全局状态污染）
        chat_model = self._create_chat_model(custom_model)
        prompt = self._create_prompt()
        tools = custom_tools or self.default_tools
        system_prompt = custom_system_prompt or self.default_system_prompt

        # 2. 创建 Agent
        agent = create_tool_calling_agent(chat_model, tools, prompt)

        # 3. 创建 Executor
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            return_intermediate_steps=return_intermediate_steps,
            **kwargs
        )


agent_factory = AgentFactory()


def get_agent_executor():
    return agent_factory.create_agent_executor()


async def get_agent_stream_response(
        query: str,
        session_id: str,
        user_id: str,
):
    thinking_queue = asyncio.Queue()
    agent_result_holder = {"response": None, "error": None}
    agent_done = asyncio.Event

    async def thinking_callback(data: dict):
        await thinking_queue.put(data)

    async def run_agent():

        try:
            set_current_user_id(user_id)
            set_thinking_callback(thinking_callback())

            history = session_manager.get_history(session_id, user_id)

            chat_history: List[BaseMessage] = []

            if history:
                for user_msg, assistant_msg in history:
                    chat_history.append(HumanMessage(content=user_msg))
                    chat_history.append(AIMessage(content=assistant_msg))

            agent_executor = agent_factory.create_agent_executor()

            full_response = []

            async for chunk in agent_executor.astream(
                    {
                        "input":query,
                        "chat_history":chat_history,
                        "system_prompt":agent_factory.default_system_prompt
                    }
            ):
                if "output" in chunk:
                    full_response.append(chunk["output"])
                elif "intermediate_steps" in chunk:
                    for action,observation  in chunk["intermediate_steps"]:
                        logger.info(f"\n\n🧠 [Agent 思考] {action.log}")
                        logger.info(f"🛠️ [调用工具] {action.tool}")
                        logger.info(f"📥 [工具输入] {action.tool_input}")
                        logger.info(f"📤 [工具结果] {observation}\n")

                agent_result_holder["response"] = "".join(full_response) if full_response else "抱歉，我无法理解您的请求。"
        finally:
            agent_done.set()

    agent_task = asyncio.create_task(run_agent())

    yield f"agent已建立连接"


    while not agent_done.is_set():
        pass


