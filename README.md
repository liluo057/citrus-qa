# 🍊 柑橘黄龙病智能问答系统 (Citrus HLB QA System)

基于 RAG（检索增强生成）技术的柑橘黄龙病专业问答系统，整合 29 篇中英文文献与科普资料，为科研人员和产业一线提供精准、可溯源的智能问答服务。

## 技术架构

- **前端**
: Gradio Web 界面
- **向量数据库**
: Chroma (本地持久化)
- **Embedding**
: 智谱 Embedding-3
- **LLM**
: DeepSeek-Chat
- **开发环境**
: WSL + Conda (Python 3.11)

## 快速开始

```bash
conda activate citrus
cd src
python build_kb.py  # 构建知识库
python app.py       # 启动 Web 界面
访问 http://localhost:7860

## 项目结构

citrus-qa/
├── data/raw/          # 29篇文献
├── src/               # 核心代码
│   ├── build_kb.py    # 知识库构建
│   ├── ask.py         # 终端问答
│   └── app.py         # Web 界面
└── requirements.txt   # 依赖清单
