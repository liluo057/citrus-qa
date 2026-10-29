"""
混合检索模块：向量检索 + BM25 关键词检索 + RRF 融合
ask.py 和 app.py 都从这里调用 hybrid_search()，改检索策略只需要改这一个文件。
"""
import os
import jieba
import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings

# 加载环境变量
load_dotenv()
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# ========== 配置 ==========
CHROMA_DIR = os.path.expanduser("~/citrus-qa/data/chroma")
COLLECTION_NAME = "citrus_hlb"

VECTOR_TOP_K = 20  # 向量检索先召回20条
BM25_TOP_K = 20    # 关键词检索先召回20条
FINAL_TOP_K = 3    # 融合后取前3条给大模型
RRF_K = 60         # RRF 融合公式的平滑常数，一般用60即可

# ========== 初始化 Chroma ==========
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection(name=COLLECTION_NAME)

# ========== 启动时把全部文本块读出来，构建 BM25 索引 ==========
# collection.get() 不传 id 就是取出所有数据
_all = collection.get(include=["documents", "metadatas"])
IDS = _all["ids"]            # 每个块的唯一编号
DOCS = _all["documents"]     # 每个块的文本
METAS = _all["metadatas"]    # 每个块的来源信息
ID2IDX = {doc_id: i for i, doc_id in enumerate(IDS)}  # 编号 → 位置，方便反查

print(f"📚 已加载 {len(IDS)} 个文本块，正在构建 BM25 关键词索引...")
# BM25 是英文算法，中文要先用 jieba 分词
_tokenized_docs = [list(jieba.cut(doc)) for doc in DOCS]
BM25 = BM25Okapi(_tokenized_docs)
print("✅ BM25 索引构建完成")


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


# ========== 第一路：向量检索（擅长语义理解）==========
def vector_search(query, top_k=VECTOR_TOP_K):
    """
    返回按相似度排好序的文本块编号列表
    """
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []

    n = min(top_k, len(IDS))  # 防止库里的块比 top_k 还少
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n
    )
    return results["ids"][0]


# ========== 第二路：BM25 关键词检索（擅长专业术语、名称精确匹配）==========
def bm25_search(query, top_k=BM25_TOP_K):
    """
    返回按关键词匹配得分排好序的文本块编号列表
    """
    query_tokens = list(jieba.cut(query))
    scores = BM25.get_scores(query_tokens)  # 每个块一个得分

    # 按得分从高到低排序，取得分大于0的前 top_k 个
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [IDS[i] for i in ranked[:top_k] if scores[i] > 0]


# ========== RRF 融合：把两路检索的排名合并成一个总排名 ==========
def rrf_fuse(vector_ids, bm25_ids):
    """
    RRF（倒数排名融合）：某块在每一路里的排名越靠前，贡献的分数越高。
    只看排名、不看原始分数，所以两种不同算法的分数可以直接合并。
    """
    scores = {}
    channels = {}  # 记录每块是被哪路检索找到的，方便展示

    for channel_name, id_list in (("向量", vector_ids), ("关键词", bm25_ids)):
        for rank, doc_id in enumerate(id_list):
            # rank 从 0 开始，第1名得 1/(60+1)，第2名得 1/(60+2)……
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
            channels.setdefault(doc_id, set()).add(channel_name)

    # 按融合得分从高到低排序
    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return ranked_ids, scores, channels


# ========== 对外接口：混合检索 ==========
def hybrid_search(query, top_k=FINAL_TOP_K):
    """
    输入问题，返回融合后的 top_k 条资料。
    每条包含：text（内容）、source（来源）、score（融合得分）、channels（来自哪路检索）
    """
    vector_ids = vector_search(query)
    bm25_ids = bm25_search(query)

    ranked_ids, scores, channels = rrf_fuse(vector_ids, bm25_ids)

    retrieved = []
    for doc_id in ranked_ids[:top_k]:
        i = ID2IDX[doc_id]
        retrieved.append({
            "text": DOCS[i],
            "source": METAS[i]["source"],
            "score": scores[doc_id],
            "channels": "+".join(sorted(channels[doc_id]))
        })
    return retrieved
