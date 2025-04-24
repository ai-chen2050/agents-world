from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# download multimodel
# ollama pull llava
llm = ChatOllama(model="llava", temperature=0)

def prompt_func(data):
    '''构造多模态输入'''

chain = prompt_func | llm | StrOutputParser()

query_chain = chain.invoke(
    {"text": "这个图片里是什么动物啊?", "image": "image_b64"}
)