# 🍊 柑橘黄龙病智能问答系统 (Citrus HLB QA System)

基于 RAG（检索增强生成）技术的柑橘黄龙病专业问答系统，整合多篇中英文文献与科普资料，为科研人员和产业一线提供精准、可溯源的智能问答服务。

## 技术架构

- **前端**：Gradio Web 界面
- **文本切块**：按段落 / 句子切块，每块约 300 字，相邻块重叠 50 字
- **向量数据库**：Chroma（本地持久化）
- **Embedding**：智谱 Embedding-3
- **检索策略**：向量检索 + BM25 关键词检索（jieba 中文分词），RRF 融合排名
- **LLM**：DeepSeek-Chat
- **开发环境**：WSL + Conda (Python 3.11)

## 检索流程

```
用户问题
   │
   ├── 向量检索（语义匹配，召回 Top 20）
   │
   └── BM25 关键词检索（术语精确匹配，召回 Top 20）
   │
   ▼
RRF 融合排名（按两路排名合并打分）
   │
   ▼
取 Top 3 资料 + 问题 → DeepSeek 生成回答（附来源）
```

## 快速开始

```bash
conda activate citrus
pip install -r requirements.txt  # 安装依赖

cd src
python build_kb.py  # 构建知识库（切块 + 向量化）
python app.py       # 启动 Web 界面
```

访问 http://localhost:7860

也可以在终端直接问答：

```bash
python ask.py
```

> 注意：修改切块参数（`build_kb.py` 顶部的 `CHUNK_SIZE` / `CHUNK_OVERLAP`）后，
> 需要删除 `data/chroma/` 并重新运行 `build_kb.py` 重建知识库。
> 检索参数（`retriever.py` 顶部的 `VECTOR_TOP_K` / `BM25_TOP_K` / `FINAL_TOP_K`）改完重启即可生效。

## 项目结构

```
citrus-qa/
├── data/raw/          # 文献
├── src/               # 核心代码
│   ├── build_kb.py    # 知识库构建（重叠切块 + 向量化）
│   ├── retriever.py   # 混合检索（向量 + BM25 + RRF 融合）
│   ├── ask.py         # 终端问答
│   └── app.py         # Web 界面
└── requirements.txt   # 依赖清单
```
