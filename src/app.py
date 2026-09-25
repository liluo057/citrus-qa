import os
import requests
import gradio as gr
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# 加载环境变量
load_dotenv()
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 配置
CHROMA_DIR = os.path.expanduser("~/citrus-qa/data/chroma")
COLLECTION_NAME = "citrus_hlb"
TOP_K = 3

# 初始化 Chroma（全局只加载一次）
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection(name=COLLECTION_NAME)

# ========== 核心函数（和 ask.py 一样）==========

def get_embedding(text):
    """调用智谱 embedding API"""
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {"model": "embedding-3", "input": text}
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    return None

def search_knowledge_base(query, top_k=TOP_K):
    """检索知识库"""
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "text": doc,
            "source": meta["source"],
            "distance": dist
        })
    return retrieved

def generate_answer(query, retrieved_chunks):
    """调用 DeepSeek 生成回答"""
    context = "\n\n".join([
        f"【资料{i+1}】（来源：{chunk['source']}）\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    
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

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return None

# ========== Gradio 界面函数 ==========

def answer_question(query):
    """
    Gradio 调用的主函数：接收问题，返回答案 + 来源信息
    """
    if not query.strip():
        return "请输入问题", ""
    
    # 检索
    retrieved = search_knowledge_base(query)
    if not retrieved:
        return "检索失败，请检查网络或 API 配置", ""
    
    # 生成回答
    answer = generate_answer(query, retrieved)
    if not answer:
        return "生成回答失败", ""
    
    # 格式化来源信息
    sources_text = "### 📚 参考资料\n\n"
    for i, chunk in enumerate(retrieved):
        sources_text += f"**{i+1}. {chunk['source']}**\n"
        sources_text += f"- 相似度距离：{chunk['distance']:.3f}\n"
        sources_text += f"- 内容预览：{chunk['text'][:100]}...\n\n"
    
    return answer, sources_text

# ========== 创建 Gradio 界面 ==========

# 自定义 CSS（让界面更好看）
custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
"""

with gr.Blocks(css=custom_css, title="柑橘黄龙病智能问答系统") as demo:
    # 标题
    gr.Markdown("""
    # 🍊 柑橘黄龙病智能问答系统
    
    基于 RAG（检索增强生成）技术，整合 多 篇中英文文献，为您提供专业的柑橘黄龙病问答服务。
    
    **使用说明**：在下方输入您的问题，系统将从知识库中检索相关资料并生成回答。
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # 输入框
            query_input = gr.Textbox(
                label="🤔 您的问题",
                placeholder="例如：柑橘黄龙病的病原菌是什么？",
                lines=2
            )
            
            # 按钮
            submit_btn = gr.Button("🔍 查询", variant="primary", size="lg")
            
            # 回答输出
            answer_output = gr.Markdown(label="📋 回答")
        
        with gr.Column(scale=1):
            # 来源信息
            sources_output = gr.Markdown(label="📚 参考资料")
    
    # 示例问题
    gr.Examples(
        examples=[
            ["柑橘黄龙病的病原菌是什么？"],
            ["黄龙病有哪些传播途径？"],
            ["目前有哪些检测黄龙病的方法？"],
            ["AI 图像识别怎么用于黄龙病检测？"],
            ["柑橘黄龙病能治愈吗？"],
        ],
        inputs=query_input,
        label="💡 示例问题"
    )
    
    # 绑定按钮点击事件
    submit_btn.click(
        fn=answer_question,
        inputs=query_input,
        outputs=[answer_output, sources_output]
    )
    
    # 也支持按回车提交
    query_input.submit(
        fn=answer_question,
        inputs=query_input,
        outputs=[answer_output, sources_output]
    )
    
    # 页脚
    gr.Markdown("""
    ---
    **技术架构**：智谱 Embedding-3（向量化）+ Chroma（向量数据库）+ DeepSeek（生成）
    
    **项目地址**：[GitHub - citrus-qa](https://github.com/your-username/citrus-qa)（待更新）
    """)

# 启动
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)