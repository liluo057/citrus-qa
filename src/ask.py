import os
import requests
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# 加载环境变量
load_dotenv()
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not ZHIPU_API_KEY or not DEEPSEEK_API_KEY:
    print("❌ 错误：请检查 ZHIPU_API_KEY 和 DEEPSEEK_API_KEY 环境变量")
    exit(1)

# ========== 配置 ==========
CHROMA_DIR = os.path.expanduser("~/citrus-qa/data/chroma")
COLLECTION_NAME = "citrus_hlb"
TOP_K = 3  # 检索最相似的3块

# ========== 初始化 Chroma ==========
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection(name=COLLECTION_NAME)

# ========== 调用智谱 Embedding ==========
def get_embedding(text):
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {"model": "embedding-3", "input": text}
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        print(f"❌ Embedding 失败：{response.status_code}")
        return None

# ========== 检索知识库 ==========
def search_knowledge_base(query, top_k=TOP_K):
    """
    把问题向量化，在 Chroma 中找最相似的文本块
    """
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # 整理结果
    chunks = results["documents"][0]  # 文本内容
    metadatas = results["metadatas"][0]  # 来源信息
    distances = results["distances"][0]  # 相似度距离（越小越相似）
    
    retrieved = []
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        retrieved.append({
            "text": chunk,
            "source": meta["source"],
            "distance": dist
        })
    
    return retrieved

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
        print("\n🔍 正在检索知识库...")
        retrieved = search_knowledge_base(query)
        
        if not retrieved:
            print("❌ 检索失败，请检查网络或 API 配置")
            continue
        
        print(f"✅ 找到 {len(retrieved)} 条相关资料：")
        for i, chunk in enumerate(retrieved):
            print(f"   {i+1}. {chunk['source']} (相似度距离: {chunk['distance']:.3f})")
        
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