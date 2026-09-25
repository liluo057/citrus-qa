import os
import glob
import requests
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# 加载环境变量
load_dotenv()
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

if not ZHIPU_API_KEY:
    print("❌ 错误：没有找到 ZHIPU_API_KEY 环境变量")
    exit(1)

# ========== 配置 ==========
DATA_DIR = os.path.expanduser("~/citrus-qa/data/raw")
CHROMA_DIR = os.path.expanduser("~/citrus-qa/data/chroma")
COLLECTION_NAME = "citrus_hlb"
CHUNK_SIZE = 300  # 每块约300字

# ========== 文本切块 ==========
def split_text(text, chunk_size=300):
    """
    按段落切块，如果段落太长再按句子切
    """
    chunks = []
    
    # 先按空行分段落
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    for para in paragraphs:
        # 如果段落小于 chunk_size，直接加入
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # 太长就按句子切（中文按。！？切）
            sentences = []
            current = ""
            for char in para:
                current += char
                if char in '。！？；\n':
                    sentences.append(current.strip())
                    current = ""
            if current:
                sentences.append(current.strip())
            
            # 把句子组合成接近 chunk_size 的块
            buffer = ""
            for sent in sentences:
                if len(buffer) + len(sent) <= chunk_size:
                    buffer += sent
                else:
                    if buffer:
                        chunks.append(buffer)
                    buffer = sent
            if buffer:
                chunks.append(buffer)
    
    return chunks

# ========== 调用智谱 Embedding ==========
def get_embedding(text):
    """
    调用智谱 embedding-3 接口，把文本转成向量
    """
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "embedding-3",
        "input": text
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result["data"][0]["embedding"]
    else:
        print(f"❌ Embedding 调用失败：{response.status_code}")
        print(response.text)
        return None

# ========== 主流程 ==========
def main():
    print("🚀 开始构建知识库...\n")
    
    # 1. 初始化 Chroma（持久化到本地）
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # 2. 获取或创建 collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "柑橘黄龙病文献知识库"}
    )
    
    # 3. 读取所有文献
    md_files = glob.glob(os.path.join(DATA_DIR, "*.md"))
    print(f"📚 找到 {len(md_files)} 篇文献\n")
    
    all_chunks = []  # 存储所有文本块
    all_metadata = []  # 存储每块的元数据（来源文献）
    
    # 4. 切块
    for filepath in sorted(md_files):
        filename = os.path.basename(filepath)
        print(f"📖 处理：{filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = split_text(content, CHUNK_SIZE)
        print(f"   → 切成 {len(chunks)} 块")
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": filename,
                "chunk_id": i
            })
    
    print(f"\n📦 总共 {len(all_chunks)} 个文本块")
    
    # 5. 向量化（分批处理，避免请求太大）
    print("\n🔄 开始向量化...")
    embeddings = []
    batch_size = 10  # 每批10个，避免超时
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        batch_embeddings = []
        
        for text in batch:
            emb = get_embedding(text)
            if emb:
                batch_embeddings.append(emb)
            else:
                # 如果失败，用零向量占位（后面可以重试）
                batch_embeddings.append([0.0] * 2048)
        
        embeddings.extend(batch_embeddings)
        print(f"   进度：{min(i+batch_size, len(all_chunks))}/{len(all_chunks)}")
    
    # 6. 存入 Chroma
    print("\n💾 存入 Chroma 数据库...")
    
    # 生成唯一 ID
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadata
    )
    
    print(f"\n✅ 知识库构建完成！")
    print(f"   存储路径：{CHROMA_DIR}")
    print(f"   Collection：{COLLECTION_NAME}")
    print(f"   总块数：{collection.count()}")

if __name__ == "__main__":
    main()