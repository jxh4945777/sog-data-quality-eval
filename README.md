# SoG Data Quality Evaluation Toolkit

**合成数据质量评估框架 v2.0** — 面向 SFT 训练数据的多维度质量评估工具。

> 评估 Synthesis-on-Graph (SoG) 等知识图谱驱动合成方法生成的训练数据质量，
> 支持客观指标、LLM-as-Judge 打分、Pairwise 对决三种评估模式。

---

## 目录

- [特性](#特性)
- [项目结构](#项目结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [数据格式](#数据格式)
- [评估模式](#评估模式)
  - [客观指标评估](#1-客观指标评估)
  - [LLM-as-Judge](#2-llm-as-judge)
  - [Pairwise 对决](#3-pairwise-对决)
- [LLM 配置](#llm-配置)
- [CLI 完整参数](#cli-完整参数)
- [评估指标详解](#评估指标详解)
- [加权综合评分](#加权综合评分)
- [公平性保证](#公平性保证)
- [示例输出](#示例输出)
- [引用项目](#引用项目)
- [License](#license)

---

## 特性

| 模式 | 说明 | LLM 依赖 |
|------|------|---------|
| **客观指标** | 8 维度 17 个学术标准指标，完全可复现 | 无 |
| **LLM-as-Judge** | 大模型从 4 个维度对每条样本打分 (1-5) | 需要 |
| **Pairwise 对决** | 大模型直接对比两种方法的样本，统计胜率 | 需要 |

- 支持**单方法评估**和**双方法对比**
- 加权综合评分，权重基于指标对 SFT 训练质量的贡献度
- LLM 配置灵活：OpenAI API / 自定义中转站 / 本地部署 (vLLM, Ollama, llama.cpp)
- 随机种子固定，结果完全可复现

---

## 项目结构

```
sog-data-quality-eval/
├── evaluate.py           # 主评估脚本：17 个客观指标 + 报告生成 + CLI
├── llm_judge.py          # LLM-as-Judge 模块：单样本打分 + Pairwise 对决
├── config.example.yaml   # LLM 配置模板
├── requirements.txt      # Python 依赖
├── reports/              # 评估报告输出目录
│   ├── report_birenkeji.txt
│   └── report_yinlian.txt
├── LICENSE
└── README.md
```

### 模块说明

| 文件 | 功能 | 可独立运行 |
|------|------|-----------|
| `evaluate.py` | 客观指标计算、对比报告生成、CLI 入口 | `python evaluate.py --help` |
| `llm_judge.py` | LLM 质量打分、Pairwise 对决、批量评估 | `python llm_judge.py --help` |

---

## 安装

```bash
# 克隆仓库
git clone https://github.com/jxh4945777/sog-data-quality-eval.git
cd sog-data-quality-eval

# 安装依赖
pip install -r requirements.txt
```

**依赖说明：**

| 包 | 用途 | 必需 |
|----|------|------|
| `jieba` | 中文分词 | 是 |
| `nltk` | Self-BLEU 计算 | 是 |
| `numpy` | 数值计算 | 是 |
| `openai` | LLM-as-Judge API 调用 | 仅 LLM 评估 |
| `pyyaml` | 读取配置文件 | 仅使用配置文件时 |

---

## 快速开始

### 对比评估 (最常用)

```bash
python evaluate.py \
  --sog sog_data.json \
  --doc baseline.json \
  --source corpus.jsonl \
  -o report.txt
```

### 全功能评估 (客观 + LLM + Pairwise)

```bash
export LLM_API_KEY="your-key"

python evaluate.py \
  --sog sog_data.json \
  --doc baseline.json \
  --source corpus.jsonl \
  --llm-judge \
  --pairwise \
  -o report.txt
```

### 单方法评估

```bash
python evaluate.py \
  --data my_data.json \
  --source corpus.jsonl \
  --name "SoG" \
  -o report.txt
```

---

## 数据格式

### 合成数据 (JSON)

标准 SFT 训练数据格式，包含 `instruction`、`input`、`output` 三个字段：

```json
[
  {
    "instruction": "你是一个知识问答助手...",
    "input": "### 信息:\n多段信息文本...\n### 问题: 具体问题",
    "output": "<think>\n1. 分析步骤一...\n2. 分析步骤二...\n</think>\n最终回答..."
  }
]
```

**字段解析规则：**
- `input` 中 `### 信息:` 到 `### 问题:` 之间为**信息区** (context)
- `input` 中 `### 问题:` 之后为**问题** (question)
- `output` 中 `<think>...</think>` 为**推理链** (reasoning chain)
- `output` 中 `</think>` 之后为**最终回答** (answer)

### 源文档 (JSONL)

每行一个 JSON 对象，支持以下格式：

```jsonl
{"chunks": ["片段1文本", "片段2文本", "..."]}
{"text": "完整文档文本"}
```

---

## 评估模式

### 1. 客观指标评估

无需 LLM，基于文本统计和信息检索方法，完全可复现。

```bash
# 对比模式
python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl

# 单方法模式
python evaluate.py --data sog.json --source corpus.jsonl --name "SoG"
```

**输出内容：**
- 8 维度 17 指标的逐项对比
- 难度分布可视化
- 维度级别胜负统计
- 加权综合得分

### 2. LLM-as-Judge

由大模型对每条样本从 4 个维度打分 (1-5 分)：

| 维度 | 评分标准 |
|------|---------|
| **回答质量** | 完整性、准确性、是否引用具体数据和细节 |
| **推理质量** | 逻辑严密性、多步推理、多角度分析 |
| **知识利用** | 对信息区内容的引用和整合程度 |
| **训练价值** | 作为 SFT 数据对模型学习推理能力的教学价值 |

```bash
# 需先配置 LLM API (见下文)
python evaluate.py --sog sog.json --doc doc.json --llm-judge -o report.txt

# 独立运行 LLM 评估
python llm_judge.py --data sog.json --name "SoG" --n 50
```

### 3. Pairwise 对决

从两种方法各随机抽取样本，由 LLM 直接对比判断哪个更优：

```bash
python evaluate.py --sog sog.json --doc doc.json --pairwise -o report.txt

# 独立运行 Pairwise
python llm_judge.py --data sog.json --data-b doc.json --name "SoG" --name-b "Doc" --pairwise-n 30
```

**公平性设计：**
- 随机化样本呈现顺序 (A/B 位置随机交换)，消除位置偏差
- 输出统计：A 胜 / B 胜 / 平局 的次数和比例

---

## LLM 配置

支持三种配置方式（按优先级从高到低）：

### 方式 1：配置文件

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml
python evaluate.py --sog sog.json --doc doc.json --llm-judge --llm-config config.yaml
```

配置文件示例：

```yaml
api_base: "https://api.openai.com/v1"
api_key: "sk-your-key"
model: "gpt-4o-mini"
temperature: 0.1
max_retries: 3
retry_delay: 2
```

### 方式 2：环境变量 (推荐)

```bash
export LLM_API_BASE="https://api.openai.com/v1"
export LLM_API_KEY="sk-your-key"
export LLM_MODEL="gpt-4o-mini"
```

### 方式 3：OpenAI 标准变量

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-your-key"
```

### 本地模型部署

```bash
# vLLM
export LLM_API_BASE="http://localhost:8000/v1"
export LLM_API_KEY="not-needed"
export LLM_MODEL="Qwen/Qwen2.5-72B-Instruct"

# Ollama
export LLM_API_BASE="http://localhost:11434/v1"
export LLM_API_KEY="ollama"
export LLM_MODEL="qwen2.5:72b"

# llama.cpp server
export LLM_API_BASE="http://localhost:8080/v1"
export LLM_API_KEY="not-needed"
export LLM_MODEL="local-model"
```

---

## CLI 完整参数

### evaluate.py

```
python evaluate.py [OPTIONS]

数据输入:
  --sog PATH            SoG 合成数据文件 (JSON)
  --doc PATH            对比方法数据文件 (JSON)
  --data PATH           单方法数据文件 (JSON)
  --source PATH         源文档文件 (JSONL)

命名:
  --sog-name NAME       SoG 方法名称 (默认: SoG)
  --doc-name NAME       对比方法名称 (默认: Doc)
  --name NAME           单方法模式的方法名称 (默认: Method)

评估模式:
  --llm-judge           启用 LLM-as-Judge 单样本评估
  --pairwise            启用 Pairwise 对决评估
  --llm-config PATH     LLM 配置文件路径 (YAML)

输出:
  -o, --output PATH     报告输出路径 (默认: 打印到终端)
```

### llm_judge.py (独立运行)

```
python llm_judge.py [OPTIONS]

  --data PATH           数据文件 (JSON, 必需)
  --name NAME           方法名称 (默认: Method)
  --n N                 评估样本数 (默认: 20)
  --config PATH         LLM 配置文件 (YAML)

Pairwise 模式:
  --data-b PATH         对比方法数据文件 (JSON)
  --name-b NAME         对比方法名称 (默认: Baseline)
  --pairwise-n N        Pairwise 评估对数 (默认: 20)
```

---

## 评估指标详解

### 8 维度 · 17 指标

| # | 维度 | 指标 | 说明 | 方向 | 权重 | 学术出处 |
|---|------|------|------|------|------|---------|
| M01 | **A. 问答相关性** | QA Token-F1 | 回答与问题的词袋 F1 重叠度 | ↑ | 3 | — |
| M02 | **B. 忠实性** | ROUGE-L F1 | 回答对源文档信息的忠实度 | ↑ | 2 | Lin, ACL 2004 |
| M03 | **C. 推理深度与质量** | 平均推理步数 | `<think>` 内编号步骤均值，无推理链计 0 | ↑ | 3 | — |
| M04 | | 多视角推理率 | 展开 ≥3 个分析角度的推理样本占比 | ↑ | 3 | Paul & Elder 批判性思维 |
| M05 | | 推理-信息整合度 | 推理链对信息区术语的引用召回率 | ↑ | 3 | — |
| M06 | **D. 上下文构建** | 信息区句数 | 信息区平均句数，衡量上下文丰富度 | ↑ | 2 | — |
| M07 | | 实体共现数 | 信息区领域实体共现数量 | ↑ | 2 | IE 实体密度 |
| M08 | **E. 难度梯度** | 难度分布熵 | 四级难度的归一化 Shannon 熵 | ↑ | 2 | Bengio et al. 2009 |
| M09 | | 深度推理占比 | 推理步数 ≥5 步的样本比例 | ↑ | 2 | — |
| M10 | **F. 知识覆盖与规模** | 源文档 Trigram 覆盖 | 合成数据信息区覆盖源文档的程度 | ↑ | 2 | — |
| M11 | | 去重后有效规模 | output 前 80 字符去重后唯一样本数 | ↑ | 2 | — |
| M12 | | 产出率 | QA 对数 / 源文档片段数 | ↑ | 2 | — |
| M13 | **G. 上下文利用** | 实体利用率 | 信息区实体在回答中被引用的比例 | ↑ | 3 | RAGAS, Es et al. 2023 |
| M14 | | 多段落引用率 | 回答引用了信息区不同段落的比例 | ↑ | 2 | RAGAS, Es et al. 2023 |
| M15 | | 问题-上下文对齐 | 问题关键词在信息区中的覆盖率 | ↑ | 2 | RAGAS, Es et al. 2023 |
| M16 | **H. 推理连贯性** | 推理链传递性 | 后续步骤引用前步结论的比例 | ↑ | 2 | Dingo PRRC, arxiv 2504.14194 |
| M17 | | 步骤均衡度 CV | 推理步骤长度的变异系数 | ↓ | 2 | distilabel DEITA, Liu et al. ICLR 2024 |

### 指标计算方法

**QA Token-F1 (M01)**
使用 jieba 分词后，计算回答与问题之间的 token 级 Precision、Recall 和 F1。

**ROUGE-L 忠实度 (M02)**
计算回答与信息区之间的最长公共子序列 (LCS)，取 F1。

**推理步数 (M03)**
正则匹配 `<think>` 标签内的编号步骤 (`1.`, `2.`, `3、` 等)，全样本平均。

**多视角推理率 (M04)**
推理链中包含 ≥3 个编号步骤，或含有多视角标记 (`一方面…另一方面` 等) 的比例。

**推理-信息整合度 (M05)**
信息区中 ≥3 字符的中文术语在推理链中出现的召回率。

**实体共现数 (M07)**
通过后缀匹配 (公司、技术、平台、系统等) 识别领域实体，统计每条信息区中的实体数。

**难度分布熵 (M08)**
将样本按推理步数分为 4 级 (直答/简单/中等/深度)，计算归一化 Shannon 熵。

**源文档 Trigram 覆盖 (M10)**
对每个源文档片段，计算其 trigram 在合成数据信息区中出现的比例，取均值。

**实体利用率 (M13)**
参考 RAGAS Context Entity Recall，统计信息区实体在推理链 + 回答中被引用的比例。

**推理链传递性 (M16)**
参考 Dingo MetaRater PRRC 框架，检测后续步骤是否引用前步的 ≥3 字符关键词 (≥2 个)。

---

## 加权综合评分

每个指标按方向归一化为 0-1 分，乘以权重后求加权平均，映射到百分制：

```
综合得分 = Σ(归一化分数_i × 权重_i) / Σ(权重_i) × 100
```

**权重设计原则：**
- **权重 3** (核心质量): QA 相关性、推理步数、多视角推理率、推理整合度、实体利用率
- **权重 2** (重要补充): 忠实性、上下文构建、难度梯度、知识覆盖、其余指标

权重反映各指标对下游 SFT 模型推理能力训练的重要性。

---

## 公平性保证

| 措施 | 说明 |
|------|------|
| 等量采样 | `min(N_sog, N_doc, 500)`，双方使用相同数量样本 |
| 全量评估 | 知识覆盖类指标 (M10-M12) 使用全量数据 |
| 固定种子 | `random.seed(42)`，结果完全可复现 |
| 中文分词 | jieba 分词，客观统一 |
| 无 LLM 依赖 | 17 个客观指标不依赖 LLM 评分 |
| 位置随机化 | Pairwise 对决中随机交换 A/B 位置 |
| 方向标注 | 每个指标明确标注 ↑高好 或 ↓低好 |

---

## 示例输出

### 对比评估摘要

```
══════════════════════════════════════════════════════════
  综合汇总
══════════════════════════════════════════════════════════

  维度              指标                   SoG        Doc    方向    胜出
  ──────────────── ──────────────── ────── ──────  ──────  ────
  A.问答相关性       QA相关性(F1)        0.2499   0.2142   ↑高好   SoG
  B.忠实性          ROUGE-L忠实度       0.1499   0.1682   ↑高好   Doc
  C.推理深度与质量    平均推理步数           1.95     1.00   ↑高好   SoG
  C.推理深度与质量    多视角推理率(%)       67.0     28.4   ↑高好   SoG
  ...

  指标胜负: SoG 14/17   Doc 3/17
  维度级别: SoG 7  Doc 1  平局 0

  加权综合得分
  SoG 综合得分: 99.2 / 100
  Doc 综合得分: 63.4 / 100
  SoG 领先: +35.8 分
```

### LLM-as-Judge 输出

```
  ▌单样本质量评估 (各 50 条)
    回答质量      SoG=4.12  Doc=3.24  → SoG
    推理质量      SoG=4.35  Doc=2.18  → SoG
    知识利用      SoG=3.98  Doc=3.01  → SoG
    训练价值      SoG=4.28  Doc=2.65  → SoG

    综合均分      SoG=4.18  Doc=2.77
```

### Pairwise 对决输出

```
  Pairwise 对决 (30 对)
  SoG 胜: 23 (76.7%)
  Doc 胜: 4 (13.3%)
  平局:   3 (10.0%)
```

---

## 引用项目

| 项目 | 本工具中的应用 |
|------|-------------|
| [RAGAS](https://github.com/explodinggradients/ragas) | Context Entity Recall (M13), Context Utilization (M14), Context Precision (M15) |
| [Dingo](https://github.com/MigoXLab/dingo) | MetaRater PRRC — 推理链传递性 (M16) |
| [Data-Juicer](https://github.com/modelscope/data-juicer) | 数据质量评估方法论参考 |
| [distilabel](https://github.com/argilla-io/distilabel) | DEITA 质量评分 — 步骤均衡度 (M17) |
| [DataMan](https://github.com/pengr/DataMan) | 质量维度体系设计参考 |

**学术参考：**
- Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *ACL Workshop*.
- Es, S. et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv*.
- Bengio, Y. et al. (2009). Curriculum Learning. *ICML*.
- Liu, W. et al. (2024). What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning. *ICLR*.
- Paul, R. & Elder, L. (2006). Critical Thinking: The Nature of Critical and Creative Thought.
- Halliday, M. & Hasan, R. (1976). Cohesion in English. *Longman*.

---

## License

[MIT](LICENSE)
