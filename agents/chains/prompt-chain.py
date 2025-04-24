from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage

system_message = SystemMessage(content="""
你是一位经验丰富的电商文案撰写专家。你的任务是根据给定的产品信息创作吸引人的商品描述。
请确保你的描述简洁、有力，并且突出产品的核心优势。
""")

human_message_template = """
请为以下产品创作一段吸引人的商品描述：
产品类型: {product_type}
核心特性: {key_feature}
目标受众: {target_audience}
价格区间: {price_range}
品牌定位: {brand_positioning}

请提供以下三种不同风格的描述，每种大约50字：
1. 理性分析型
2. 情感诉求型
3. 故事化营销型
"""

# 示例使用
product_info_input = {
    "system_message": system_message,
    "product_type": "智能手表",
    "key_feature": "心率监测和睡眠分析",
    "target_audience": "注重健康的年轻专业人士",
    "price_range": "中高端",
    "brand_positioning": "科技与健康的完美结合"
}

prompt = ChatPromptTemplate.from_template(human_message_template)

model = OllamaLLM(model="nezahatkorkmaz/deepseek-v3")
# model = OllamaLLM(model="deepseek-r1")
# model = OllamaLLM(model="llama3.1")

chain = prompt | model

# Pass search results to the chain
result = chain.invoke(product_info_input)
print(result)