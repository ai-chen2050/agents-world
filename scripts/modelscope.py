import os
from dotenv import load_dotenv
import getpass
from langchain_modelscope import ModelScopeEndpoint

# 加载环境变量
load_dotenv(dotenv_path='.env')
# 获取 API 密钥
api = os.getenv("MODELSCOPE_SDK_TOKEN")
if not api:
    os.environ["MODELSCOPE_SDK_TOKEN"] = getpass.getpass(
        "Enter your ModelScope SDK token: "
    )

llm = ModelScopeEndpoint(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0,
    max_tokens=1024,
    timeout=60,
)

input_text = "Write a quick sort algorithm in python"

completion = llm.invoke(input_text)
print(completion)