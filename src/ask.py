import os
import requests
from dotenv import load_dotenv
from retriever import hybrid_search  # 混合检索：向量 + BM25 + RRF 融合

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    print("❌ 错误：请检查 DEEPSEEK_API_KEY 环境变量")
    exit(1)

# ========== 配置 ==========
TOP_K = 3  # 检索最相似的3块

# ========== 检索知识库 ==========
def search_knowledge_base(query, top_k=TOP_K):
    """
    混合检索：向量检索 + BM25 关键词检索，再用 RRF 融合排名
    """
    return hybrid_search(query, top_k=top_k)

# ========== 调用 DeepSeek 生成回答 ==========
def generate_answer(query, retrieved_chunks):
    """
    把检索结果 + 问题发给 DeepSeek，生成回答
    """
    # 构建上下文（检索到的资料）
    context = "\n\n".join([
        f"【资料{i+1}】（来源：{chunk['source']}）\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    ])

    # 构建 Prompt（关键：防幻觉指令）
    prompt = f"""你是一个柑橘黄龙病领域的专业问答助手。请根据以下提供的资料回答问题。

【重要规则】
1. 只能根据提供的资料回答，不要编造信息
2. 如果资料中没有相关内容，请明确说"根据现有文献，我无法回答这个问题"
3. 回答时注明信息来源（资料编号）
4. 用中文回答，语言简洁专业

【参考资料】
{context}

【用户问题】
{query}

【回答】"""

    # 调用 DeepSeek
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # 低温度，减少幻觉
        "max_tokens": 1000
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        return answer
    else:
        print(f"❌ DeepSeek 调用失败：{response.status_code}")
        print(response.text)
        return None

# ========== 主流程 ==========
def main():
    print("🍊 柑橘黄龙病智能问答系统")
    print("=" * 50)
    print("输入问题，或输入 'q' 退出\n")

    while True:
        # 获取用户输入
        query = input("🤔 你的问题：").strip()

        if query.lower() == 'q':
            print("👋 再见！")
            break

        if not query:
            continue

        # 1. 检索知识库
        print("\n🔍 正在检索知识库（向量 + 关键词混合检索）...")
        retrieved = search_knowledge_base(query)

        if not retrieved:
            print("❌ 检索失败，请检查网络或 API 配置")
            continue

        print(f"✅ 找到 {len(retrieved)} 条相关资料：")
        for i, chunk in enumerate(retrieved):
            print(f"   {i+1}. {chunk['source']} (融合得分: {chunk['score']:.4f}, 来源: {chunk['channels']})")

        # 2. 生成回答
        print("\n🤖 正在生成回答...")
        answer = generate_answer(query, retrieved)

        if answer:
            print("\n" + "=" * 50)
            print("📋 回答：")
            print(answer)
            print("=" * 50 + "\n")
        else:
            print("❌ 生成回答失败\n")

if __name__ == "__main__":
    main()
