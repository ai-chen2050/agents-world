import os
from dotenv import load_dotenv
from openai import OpenAI
import getpass

# 加载环境变量
load_dotenv(dotenv_path='.env')
# 获取 API 密钥
api = os.getenv("MODELSCOPE_SDK_TOKEN")
if not api :
    os.environ["MODELSCOPE_SDK_TOKEN"] = getpass.getpass(
        "Enter your ModelScope SDK token: "
    )


client = OpenAI(
    api_key=api, # 请替换成您的ModelScope SDK Token
    base_url="https://api-inference.modelscope.cn/v1/"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-32B-Instruct", # ModleScope Model-Id
    messages=[
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': '用python写一下快排'
        }
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end='', flush=True)