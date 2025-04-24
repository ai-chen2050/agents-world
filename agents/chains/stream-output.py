from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.1", temperature=0.7)

messages = [
    ("human", "你好呀, 请介绍一下中国的万里长城的历史背景和故事"),
]

for chunk in model.stream(messages):
    print(chunk.content, end='', flush=True)