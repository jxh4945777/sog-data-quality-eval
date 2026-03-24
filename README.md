# SoG Data Quality Evaluation Toolkit

合成数据质量评估框架 v2.0 — 面向 SFT 训练数据的多维度质量评估工具。

## 概述

本工具提供 **8 个评估维度、17 个客观指标**，加上 **LLM-as-Judge** 质量打分和 **Pairwise 对决**评估，从推理质量、上下文构建、难度梯度、知识覆盖、上下文利用等多个角度全面评估合成数据质量。

### 三大评估模式

| 模式 | 说明 | 依赖 |
|------|------|------|
| **客观指标** | 17 个学术标准指标，完全可复现 | jieba, nltk, numpy |
| **LLM-as-Judge** | 大模型从 4 个维度评分 | + openai |
| **Pairwise 对决** | 大模型直接对比两种方法 | + openai |

## 快速开始

```bash
pip install -r requirements.txt

# 对比评估 (客观指标)
python evaluate.py --sog sog_data.json --doc baseline.json --source corpus.jsonl

# 含 LLM-as-Judge
python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl --llm-judge

# 含 Pairwise 对决
python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl --pairwise

# 全部评估
python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl --llm-judge --pairwise -o report.txt

# 单方法评估
python evaluate.py --data my_data.json --source corpus.jsonl --name "SoG"
```

## LLM 配置

支持三种配置方式（按优先级）：

### 1. 配置文件
```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml 填写 API 信息
python evaluate.py --sog sog.json --doc doc.json --llm-judge --llm-config config.yaml
```

### 2. 环境变量
```bash
export LLM_API_BASE="https://api.openai.com/v1"
export LLM_API_KEY="sk-your-key"
export LLM_MODEL="gpt-4o-mini"
python evaluate.py --sog sog.json --doc doc.json --llm-judge
```

### 3. OpenAI 标准变量
```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-your-key"
```

### 本地部署
```bash
# vLLM
export LLM_API_BASE="http://localhost:8000/v1"
export LLM_API_KEY="not-needed"
export LLM_MODEL="qwen2.5-72b-instruct"

# Ollama
export LLM_API_BASE="http://localhost:11434/v1"
export LLM_API_KEY="ollama"
export LLM_MODEL="qwen2.5:72b"
```

## 数据格式

**合成数据** (JSON):
```json
[{"instruction": "...", "input": "### 信息:\n...\n### 问题: ...", "output": "<think>...</think>..."}]
```

**源文档** (JSONL):
```jsonl
{"chunks": ["片段1", "片段2", ...]}
```

## 8 维度 · 17 指标

| 维度 | 指标 | 说明 | 方向 | 权重 | 学术出处 |
|------|------|------|------|------|---------|
| **A. 问答相关性** | QA Token-F1 | 回答与问题的词袋 F1 | ↑ | 3 | — |
| **B. 忠实性** | ROUGE-L F1 | 回答对源文档的忠实度 | ↑ | 2 | Lin 2004 |
| **C. 推理深度与质量** | 平均推理步数 | `<think>` 内编号步骤均值 | ↑ | 3 | — |
| | 多视角推理率 | ≥3 个分析角度的推理占比 | ↑ | 3 | Paul & Elder |
| | 推理-信息整合度 | 推理链对信息区术语的召回率 | ↑ | 3 | — |
| **D. 上下文构建** | 信息区句数 | 上下文信息丰富程度 | ↑ | 2 | — |
| | 实体共现数 | 多实体共现=多知识源融合 | ↑ | 2 | IE 实体密度 |
| **E. 难度梯度** | 难度分布熵 | 四级难度的归一化 Shannon 熵 | ↑ | 2 | Bengio 2009 |
| | 深度推理占比 | ≥5 步推理的样本比例 | ↑ | 2 | — |
| **F. 知识覆盖** | 源文档覆盖 | Trigram Recall | ↑ | 2 | — |
| | 去重后规模 | 唯一样本数 | ↑ | 2 | — |
| | 产出率 | QA/源片段 | ↑ | 2 | — |
| **G. 上下文利用** | 实体利用率 | Context Entity Recall | ↑ | 3 | RAGAS 2023 |
| | 多段落引用率 | Context Utilization | ↑ | 2 | RAGAS 2023 |
| | 问题-上下文对齐 | Context Precision 近似 | ↑ | 2 | RAGAS 2023 |
| **H. 推理连贯性** | 推理链传递性 | 步骤间引用前步结论的比例 | ↑ | 2 | Dingo PRRC |
| | 步骤均衡度CV | 步骤长度变异系数 | ↓ | 2 | distilabel DEITA |

### LLM-as-Judge 评估维度

| 维度 | 说明 |
|------|------|
| 回答质量 | 完整性、准确性、引用具体数据 |
| 推理质量 | 逻辑严密性、多角度分析、层层递进 |
| 知识利用 | 信息区内容的利用程度 |
| 训练价值 | 对 SFT 模型学习推理能力的教学价值 |

### Pairwise 对决

随机抽取样本对，由 LLM 直接对比两种方法生成的样本质量，输出胜率统计。采用随机化顺序避免位置偏差。

## 公平性保证

- 等量采样 `min(N1, N2, 500)`，知识覆盖类使用全量
- 随机种子 42，结果可复现
- jieba 中文分词，客观指标无 LLM 依赖
- Pairwise 对决随机化样本顺序，消除位置偏差
- 加权综合得分基于指标对 SFT 训练数据质量的贡献度

## 引用项目

| 项目 | 贡献 |
|------|------|
| [RAGAS](https://github.com/explodinggradients/ragas) | Context Entity Recall, Context Precision, Context Utilization |
| [Dingo](https://github.com/MigoXLab/dingo) | MetaRater PRRC Reasoning |
| [Data-Juicer](https://github.com/modelscope/data-juicer) | 评估方法论参考 |
| [distilabel](https://github.com/argilla-io/distilabel) | DEITA 质量评分参考 |
| [DataMan](https://github.com/pengr/DataMan) | 质量维度设计参考 |

## License

MIT
