import os
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量读取 key
api_key = os.getenv("ZHIPU_API_KEY")

if not api_key:
    print("❌ 错误：没有找到 ZHIPU_API_KEY 环境变量")
    print("请检查 ~/.bashrc 是否配置正确，并执行 source ~/.bashrc")
    exit(1)

# 智谱 embedding 接口地址
url = "https://open.bigmodel.cn/api/paas/v4/embeddings"

# 请求头
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 请求体：要向量化的文本
data = {
    "model": "embedding-3",  # 使用的模型
    "input": "柑橘黄龙病是一种由韧皮部杆菌引起的毁灭性病害"
}

# 发送请求
print("正在调用智谱 embedding API...")
response = requests.post(url, headers=headers, json=data)

# 检查响应
if response.status_code == 200:
    result = response.json()
    embedding = result["data"][0]["embedding"]
    print(f"✅ 调用成功！")
    print(f"向量维度：{len(embedding)}")
    print(f"前5个数值：{embedding[:5]}")
    print(f"消耗 tokens：{result['usage']['total_tokens']}")
else:
    print(f"❌ 调用失败，状态码：{response.status_code}")
    print(f"错误信息：{response.text}")