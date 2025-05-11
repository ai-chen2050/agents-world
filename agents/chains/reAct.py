""" 
ReAct (Reasoning and Acting)
* 参考资料
* [1] https://react-lm.github.io/
* [2] https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/
"""

from langchain_core.tools import tool  
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import render_text_description
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.chat_models import ChatOllama

class SearchInput(BaseModel):
    location: str = Field(description="location to search for")  # 定义一个 Pydantic 模型，用于描述输入模式，并提供描述信息

@tool(args_schema=SearchInput)
def weather_forecast(location: str):
    """天气预报工具。"""
    print(f"Weather for {location}")  # 打印要预报天气的位置
    return f"A dummy forecast for {location}"  # 返回给定位置的虚拟天气预报

llm = ChatOllama(model="gemma:2b")  # 初始化 ChatOllama 模型，使用 "gemma:2b"
tools = [weather_forecast]  # 使用 weather_forecast 工具
prompt = hub.pull("hwchase17/react-json")  # 从 hub 拉取特定提示
prompt = prompt.partial(
    tools=render_text_description(tools),  # 为提示呈现工具的文本描述
    tool_names=", ".join([t.name for t in tools]),  # 将工具名称连接成一个以逗号分隔的字符串
)
agent = create_react_agent(llm, tools, prompt)  # 使用 llm、工具和自定义提示创建代理
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=False, format="json")  # 使用指定参数初始化 AgentExecutor
print(agent_executor.invoke({"input":"What is the weather in Paris?"}))  # 使用测试输入调用代理并打印结果

# 使用对话历史
# 拉去特定提示，注意此处使用的是 react-chat
prompt = hub.pull("hwchase17/react-chat")

# 构建 ReAct agent
agent_history = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent_history, tools=tools, verbose=False)

agent_executor.invoke(
    {
        "input": "what's my name? Only use a tool if needed, otherwise respond with Final Answer",
        "chat_history": "Human: Hi! My name is Bob\nAI: Hello Bob! Nice to meet you",
    }
)