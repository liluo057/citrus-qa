import os
from openai import OpenAI

# 从环境变量读 key，安全
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit("没检测到 DEEPSEEK_API_KEY，检查 ~/.bashrc 配置")

# DeepSeek 兼容 OpenAI 的调用格式，只需改 base_url
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一位柑橘病害专家，回答简洁专业。"},
        {"role": "user", "content": "用一句话介绍柑橘黄龙病。"},
    ],
    temperature=0.3,
)

print(response.choices[0].message.content)