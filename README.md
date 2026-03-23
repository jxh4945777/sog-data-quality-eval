# SoG Data Quality Evaluation Toolkit

合成数据质量评估框架 — 面向 SFT (Supervised Fine-Tuning) 训练数据的多维度质量评估工具。

## 概述

本工具提供 **10 个评估维度、24 个指标**，用于评估和对比不同合成数据方案的质量。所有指标均基于学术标准方法，不依赖任何 LLM 评分，完全可复现。

支持两种使用模式：
- **对比模式**：对比两种合成方法（如 SoG vs Baseline）
- **单独模式**：独立评估一种合成方法的数据质量

## 快速开始

```bash
# 安装依赖
pip install jieba nltk numpy

# 对比评估两种方法
python evaluate.py \
  --sog sog_data.json \
  --doc baseline_data.json \
  --source corpus.jsonl \
  -o report.txt

# 单独评估一种方法
python evaluate.py \
  --data my_data.json \
  --source corpus.jsonl \
  --name "MyMethod"
```

## 数据格式

### 合成数据 (JSON)

```json
[
  {
    "instruction": "请你根据给定的信息...",
    "input": "### 信息:\n...\n### 问题: ...",
    "output": "<think>\n1. 首先...\n</think>\n<answer>...</answer>"
  }
]
```

### 源文档 (JSONL)

```jsonl
{"chunks": ["文档片段1", "文档片段2", ...]}
```

## 评估维度与指标

### 10 个评估维度

| 维度 | 指标数 | 衡量内容 | 学术出处 |
|------|--------|---------|---------|
| A. 多样性 | 2 | 词汇多样性与样本间差异 | Distinct-N (NAACL'16), Self-BLEU (SIGIR'18) |
| B. 忠实性 | 2 | 答案对源文档和问题的忠实度 | ROUGE-L (ACL'04) |
| C. 信息密度 | 2 | 输出内容的信息紧凑程度 | gzip Compression (SemDeDup/D4) |
| D. 推理深度 | 2 | 推理链的步数与结论完整性 | — |
| E. 推理质量 | 3 | 多视角分析、逻辑衔接、信息整合 | Halliday & Hasan'76, Paul & Elder |
| F. 上下文构建 | 2 | 信息区丰富度与实体融合 | IE Entity Density |
| G. 难度梯度 | 2 | 难度分布均匀性与深度推理占比 | Curriculum Learning (Bengio'09) |
| H. 知识覆盖 | 3 | 源文档利用广度与数据规模 | Trigram Recall |
| I. 上下文利用 | 3 | 上下文实体利用与问题对齐 | RAGAS (Es et al.'23) |
| J. 推理连贯性 | 3 | 推理链传递性与样本内重复 | Dingo PRRC, Data-Juicer |

### 24 个指标详解

#### A. 多样性 (Diversity)

| 指标 | 公式 | 方向 | 说明 |
|------|------|------|------|
| **Distinct-2** | `\|unique bigrams\| / \|total bigrams\|` | ↑高好 | 去除结构标签和停用词后，内容词 bigram 多样性 |
| **Self-BLEU-4** | `(1/N) × Σ BLEU(s_i, S\{s_i})` | ↓低好 | 样本间 BLEU-4 均值，越低越多样 |

> 出处: Li et al., "A Diversity-Promoting Objective Function for Neural Conversation Models", NAACL 2016; Zhu et al., "Texygen", SIGIR 2018

#### B. 忠实性 (Faithfulness)

| 指标 | 公式 | 方向 | 说明 |
|------|------|------|------|
| **ROUGE-L** | LCS-based F1 | ↑高好 | 回答内容与输入信息区的最长公共子序列 |
| **QA Token-F1** | Token-level F1 | ↑高好 | 回答内容与问题的词袋重叠度 |

> 出处: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries", ACL 2004

#### C. 信息密度 (Information Density)

| 指标 | 公式 | 方向 | 说明 |
|------|------|------|------|
| **Compression Ratio** | `len(raw) / len(gzip(raw))` | ↓低好 | 越高越冗余 |
| **内容词密度** | `unique_content_tokens / chars × 100` | ↑高好 | 每单位文本中实质信息量 |

> 出处: SemDeDup, D4 等数据质量研究

#### D. 推理深度 (Reasoning Depth)

| 指标 | 方向 | 说明 |
|------|------|------|
| **平均推理步数** | ↑高好 | `<think>` 标签内编号步骤数均值，无推理链计 0 |
| **结论完整率** | ↑高好 | 推理类样本中有明确结论标记的比例 |

#### E. 推理质量 (Reasoning Quality)

| 指标 | 方向 | 说明 |
|------|------|------|
| **多视角推理率** | ↑高好 | 推理中展开 ≥3 个分析角度的比例 |
| **逻辑连接密度** | ↑高好 | think 链中因果/转折/递进连接词频率 |
| **推理-信息整合度** | ↑高好 | 推理链对信息区术语的引用召回率 |

> 出处: Paul & Elder 批判性思维框架; Halliday & Hasan (1976) 语篇衔接理论; Schiffrin (1987) 话语标记研究

#### F. 上下文构建 (Context Construction)

| 指标 | 方向 | 说明 |
|------|------|------|
| **信息区句数** | ↑高好 | 信息区平均句子数 |
| **实体共现数** | ↑高好 | 信息区中领域实体的平均共现数量 |

> 出处: 信息抽取 (IE) 中实体密度为标准质量指标

#### G. 难度梯度 (Difficulty Distribution)

| 指标 | 方向 | 说明 |
|------|------|------|
| **难度分布熵** | ↑高好 | 四级难度 (直答/简单/中等/深度) 的归一化 Shannon 熵 |
| **深度推理占比** | ↑高好 | 推理步数 ≥5 的样本比例 |

> 出处: Bengio et al. (2009) Curriculum Learning — 均匀难度分布有助于模型渐进学习

#### H. 知识覆盖 (Knowledge Coverage)

| 指标 | 方向 | 说明 |
|------|------|------|
| **源文档覆盖** | ↑高好 | 源文档 trigram 被合成数据覆盖的平均比例 (全量数据) |
| **去重后规模** | ↑高好 | output 前 80 字符去重后的有效样本数 |
| **产出率** | ↑高好 | QA 对数 / 源文档片段数 |

#### I. 上下文利用 (Context Utilization) — RAGAS

| 指标 | 方向 | 说明 |
|------|------|------|
| **实体利用率** | ↑高好 | 信息区领域实体在回答中被引用的比例 |
| **多段落引用率** | ↑高好 | 回答引用了信息区多少不同段落的信息 |
| **问题-上下文对齐** | ↑高好 | 问题关键词在信息区中的覆盖率 |

> 出处: RAGAS (Es et al., 2023) — Context Entity Recall, Context Utilization, Context Precision 的无 LLM 近似实现

#### J. 推理连贯性 (Reasoning Coherence) — Dingo/Data-Juicer

| 指标 | 方向 | 说明 |
|------|------|------|
| **推理链传递性** | ↑高好 | 后续步骤引用前步结论的比例 |
| **样本内重复率** | ↓低好 | 单条样本内 4-gram 重复比例 |
| **步骤均衡度** | ↓低好 | 各推理步骤长度的变异系数 |

> 出处: Dingo MetaRater PRRC 框架 (arxiv 2504.14194); Data-Juicer word_repetition_filter; distilabel DEITA (ICLR 2024)

## 公平性保证

- **等量采样**: A-G, I-J 维度双方均采样 `min(N_sog, N_doc, 500)` 条
- **全量数据**: H 维度 (知识覆盖) 使用全量数据，衡量方法规模能力
- **随机种子**: 固定 seed=42，结果可复现
- **中文分词**: 统一使用 jieba
- **无 LLM 依赖**: 所有指标纯统计计算，无 API 调用

## CLI 参数

```
用法: python evaluate.py [选项]

对比模式:
  --sog PATH          SoG 合成数据 (JSON)
  --doc PATH          Baseline 合成数据 (JSON)
  --sog-name NAME     SoG 方法名称 (默认: SoG)
  --doc-name NAME     Baseline 方法名称 (默认: Doc)

单独模式:
  --data PATH         合成数据路径 (JSON)
  --name NAME         方法名称 (默认: Method)

通用:
  --source PATH       源文档 (JSONL)，用于计算知识覆盖
  -o, --output PATH   输出路径 (默认: 标准输出)
```

## 引用项目

本框架整合了以下开源项目的评估方法论：

| 项目 | 贡献的指标 |
|------|----------|
| [RAGAS](https://github.com/explodinggradients/ragas) | Context Entity Recall, Context Utilization, Context Precision |
| [Dingo](https://github.com/MigoXLab/dingo) | MetaRater PRRC Reasoning 维度 |
| [Data-Juicer](https://github.com/modelscope/data-juicer) | word_repetition_filter, DiversityAnalysis |
| [distilabel](https://github.com/argilla-io/distilabel) | DEITA ComplexityScorer / QualityScorer |
| [DataMan](https://github.com/pengr/DataMan) | 质量评估维度设计参考 |

## License

MIT
