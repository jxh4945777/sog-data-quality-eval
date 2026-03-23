# SoG Data Quality Evaluation Toolkit

合成数据质量评估框架 — 面向 SFT 训练数据的多维度质量评估工具。

## 概述

本工具提供 **7 个评估维度、13 个指标**，从推理质量、上下文构建、难度梯度、知识覆盖、上下文利用等多个角度评估合成数据质量。所有指标基于学术标准方法，不依赖 LLM 评分，完全可复现。

支持两种模式：
- **对比模式**：对比两种合成方法
- **单独模式**：独立评估一种方法

## 快速开始

```bash
pip install jieba nltk numpy

# 对比评估
python evaluate.py --sog sog_data.json --doc baseline.json --source corpus.jsonl

# 单独评估
python evaluate.py --data my_data.json --source corpus.jsonl --name "SoG"
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

## 7 维度 · 13 指标

| 维度 | 指标 | 说明 | 学术出处 |
|------|------|------|---------|
| **A. 问答相关性** | QA Token-F1 | 回答与问题的词袋 F1 重叠度 | — |
| **B. 推理深度与质量** | 平均推理步数 | `<think>` 内编号步骤均值 | — |
| | 多视角推理率 | ≥3 个分析角度的推理占比 | Paul & Elder 批判性思维 |
| | 推理-信息整合度 | 推理链对信息区术语的召回率 | — |
| **C. 上下文构建** | 信息区句数 | 上下文信息丰富程度 | — |
| | 实体共现数 | 多实体共现=多知识源融合 | IE 实体密度指标 |
| **D. 难度梯度** | 难度分布熵 | 四级难度的归一化 Shannon 熵 | Bengio et al. 2009 Curriculum Learning |
| | 深度推理占比 | ≥5 步推理的样本比例 | — |
| **E. 知识覆盖与规模** | 去重后有效规模 | 去重后唯一样本数 | — |
| | 产出率 | QA 对数 / 源文档片段数 | — |
| **F. 上下文利用** | 实体利用率 | 信息区实体在回答中被引用的比例 | RAGAS Context Entity Recall (Es et al. 2023) |
| | 问题-上下文对齐 | 问题关键词在信息区的覆盖率 | RAGAS Context Precision (Es et al. 2023) |
| **G. 推理连贯性** | 推理链传递性 | 后续步骤引用前步结论的比例 | Dingo PRRC (arxiv 2504.14194); Halliday & Hasan 1976 |

## 公平性保证

- 等量采样 `min(N1, N2, 500)`，知识覆盖类使用全量
- 随机种子 42，结果可复现
- jieba 中文分词，无 LLM 依赖

## 引用项目

| 项目 | 贡献 |
|------|------|
| [RAGAS](https://github.com/explodinggradients/ragas) | Context Entity Recall, Context Precision |
| [Dingo](https://github.com/MigoXLab/dingo) | MetaRater PRRC Reasoning |
| [Data-Juicer](https://github.com/modelscope/data-juicer) | 评估方法论参考 |
| [distilabel](https://github.com/argilla-io/distilabel) | DEITA 质量评分参考 |
| [DataMan](https://github.com/pengr/DataMan) | 质量维度设计参考 |

## License

MIT
