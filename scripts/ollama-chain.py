from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="nezahatkorkmaz/deepseek-v3")

chain = prompt | model

result = chain.invoke({"question": "What is LangChain?"})
print(result)