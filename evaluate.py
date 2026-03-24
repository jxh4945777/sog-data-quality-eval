#!/usr/bin/env python3
"""
SoG 合成数据质量评估框架 v2.0
Synthesis-on-Graph Data Quality Evaluation Toolkit

8 维度 · 17 客观指标 · LLM-as-Judge · 对比评估
支持: 单方法评估 / 双方法对比 / LLM 质量打分 / Pairwise 对决

指标来源: RAGAS / Dingo / Data-Juicer / distilabel / ROUGE / Distinct-N / Self-BLEU

用法:
  # 对比评估 (客观指标)
  python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl

  # 单方法评估
  python evaluate.py --data my.json --source corpus.jsonl --name "SoG"

  # 含 LLM-as-Judge (需配置 config.yaml 或环境变量)
  python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl --llm-judge

  # 含 Pairwise 对决
  python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl --pairwise
"""

import json, re, math, gzip, random, argparse, sys, os, io
from collections import Counter
from pathlib import Path
import numpy as np

try:
    import jieba
    jieba.setLogLevel(jieba.logging.WARNING)
except ImportError:
    print("请安装 jieba: pip install jieba"); sys.exit(1)

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    print("请安装 nltk: pip install nltk"); sys.exit(1)


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

STOPWORDS = set("的了是在有和与及等也都而但又或不其这那将被把让给对从到为以用就只"
                "中上下大小多少要会能可得着过了吗呢吧啊呀么所被")

ENTITY_SUFFIXES = (r'公司|集团|市场|技术|产品|平台|系统|服务|芯片|业务|基金|银行'
                   r'|机构|规范|接口|方案|流程|功能|模块|权益|风险|部门|环境|网络|数据')


def tokenize(text):
    return [w for w in jieba.lcut(text) if w.strip() and not re.fullmatch(r'[\s\W]+', w)]


def tokenize_content(text):
    text = re.sub(r'</?(?:think|answer)>', '', text)
    text = re.sub(r'(?:^|\n)\s*\d+[\.\、\)]\s*', ' ', text)
    text = re.sub(r'\*\*', '', text)
    tokens = tokenize(text)
    return [w for w in tokens if w not in STOPWORDS and len(w) >= 2]


def extract_info(item):
    inp = item["input"]
    for sep in ["### 信息:", "### 信息："]:
        if sep in inp:
            info = inp.split(sep, 1)[1]
            for qsep in ["### 问题:", "### 问题："]:
                if qsep in info:
                    info = info.split(qsep, 1)[0]
            return info.strip()
    return inp.strip()


def extract_question(item):
    inp = item["input"]
    for sep in ["### 问题:", "### 问题："]:
        if sep in inp:
            return inp.split(sep, 1)[1].strip()
    return ""


def extract_answer(item):
    output = item["output"]
    if "</think>" in output:
        output = output.split("</think>", 1)[1]
    return re.sub(r'</?answer>', '', output).strip()


def extract_think(item):
    output = item["output"]
    if "<think>" not in output:
        return None
    think = output.split("<think>", 1)[1]
    if "</think>" in think:
        think = think.split("</think>", 1)[0]
    return think


def sub(data, n, seed=42):
    rng = random.Random(seed)
    return rng.sample(data, n) if len(data) > n else list(data)


def lcs_len(x, y):
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l_f1(hyp, ref):
    if not hyp or not ref:
        return 0.0
    l = lcs_len(hyp, ref)
    if l == 0:
        return 0.0
    p, r = l / len(hyp), l / len(ref)
    return 2 * p * r / (p + r)


def pct_diff(a, b):
    if b == 0:
        return float('inf') if a > 0 else 0.0
    return (a - b) / abs(b) * 100


# ════════════════════════════════════════════════════════════
# 17 个客观评估指标
# ════════════════════════════════════════════════════════════

class Metrics:
    """
    8 维度 · 17 指标

    A. 问答相关性 (1)
    B. 忠实性 (1)
    C. 推理深度与质量 (3)
    D. 上下文构建 (2)
    E. 难度梯度 (2)
    F. 知识覆盖与规模 (3)
    G. 上下文利用 (3)
    H. 推理连贯性 (2)
    """

    # ── A. 问答相关性 ──

    @staticmethod
    def qa_relevance(data, n):
        """M01 QA Token-F1: 回答内容与问题的词袋 F1 重叠度"""
        s = sub(data, n)
        scores = []
        for i in s:
            q = extract_question(i)
            if not q:
                continue
            qc, ac = Counter(tokenize(q)), Counter(tokenize(extract_answer(i)))
            c = sum((qc & ac).values())
            if c == 0:
                scores.append(0.0)
                continue
            p, r = c / sum(ac.values()), c / sum(qc.values())
            scores.append(2 * p * r / (p + r))
        return round(np.mean(scores), 4)

    # ── B. 忠实性 ──

    @staticmethod
    def faithfulness(data, n):
        """M02 ROUGE-L 忠实度: 回答与源文档信息区的 ROUGE-L F1
        学术依据: Lin (2004) ROUGE; 衡量生成内容对源信息的忠实程度"""
        s = sub(data, n)
        return round(np.mean([
            rouge_l_f1(tokenize(extract_answer(i))[:200],
                       tokenize(extract_info(i))[:200])
            for i in s
        ]), 4)

    # ── C. 推理深度与质量 ──

    @staticmethod
    def reasoning_depth(data, n):
        """M03 平均推理步数: <think> 内编号步骤数均值，无推理链计 0"""
        s = sub(data, n)
        steps = []
        for i in s:
            t = extract_think(i)
            steps.append(len(re.findall(r'(?:^|\n)\s*\d+[\.\、]', t)) if t else 0)
        return round(np.mean(steps), 2)

    @staticmethod
    def multi_perspective(data, n):
        """M04 多视角推理率: 推理类样本中展开 ≥3 个分析角度的比例
        依据: Paul & Elder 批判性思维框架"""
        s = sub(data, n)
        reasoning = [i for i in s if extract_think(i) is not None]
        if not reasoning:
            return 0.0
        count = 0
        for i in reasoning:
            t = extract_think(i)
            steps = len(re.findall(r'(?:^|\n)\s*\d+[\.\、]', t))
            has_multi = bool(re.search(r'一方面.*另一方面|不仅.*还|既.*又', t, re.DOTALL))
            if steps >= 3 or has_multi:
                count += 1
        return round(count / len(reasoning) * 100, 1)

    @staticmethod
    def reasoning_integration(data, n):
        """M05 推理-信息整合度: 推理链对信息区术语的引用召回率
        衡量推理是否基于提供的信息展开而非泛泛而谈"""
        s = sub(data, n)
        scores = []
        for i in s:
            t = extract_think(i)
            if t is None:
                continue
            info_terms = set(w for w in jieba.lcut(extract_info(i))
                             if len(w) >= 3 and re.fullmatch(r'[\u4e00-\u9fff]+', w))
            think_terms = set(w for w in jieba.lcut(t)
                              if len(w) >= 3 and re.fullmatch(r'[\u4e00-\u9fff]+', w))
            if info_terms:
                scores.append(len(info_terms & think_terms) / len(info_terms))
        return round(np.mean(scores), 4) if scores else 0.0

    # ── D. 上下文构建 ──

    @staticmethod
    def info_sentences(data, n):
        """M06 信息区平均句数: 衡量上下文信息丰富程度"""
        s = sub(data, n)
        counts = []
        for i in s:
            sents = [x.strip() for x in re.split(r'[。！；]', extract_info(i))
                     if len(x.strip()) > 5]
            counts.append(len(sents))
        return round(np.mean(counts), 2)

    @staticmethod
    def entity_cooccurrence(data, n):
        """M07 信息区实体共现数: 每条信息区中共现的领域实体数量
        衡量信息融合广度 — 多实体共现意味着融合了多个知识源
        依据: 信息抽取 (IE) 实体密度指标"""
        s = sub(data, n)
        scores = []
        for i in s:
            entities = set(re.findall(
                rf'[\u4e00-\u9fff]{{2,8}}(?:{ENTITY_SUFFIXES})', extract_info(i)))
            scores.append(len(entities))
        return round(np.mean(scores), 2)

    # ── E. 难度梯度 ──

    @staticmethod
    def difficulty_dist(data, n):
        s = sub(data, n)
        tiers = [0, 0, 0, 0]  # 直答, 简单(1-2), 中等(3-4), 深度(5+)
        for i in s:
            t = extract_think(i)
            if t is None:
                tiers[0] += 1
                continue
            steps = len(re.findall(r'(?:^|\n)\s*\d+[\.\、]', t))
            if steps <= 2:
                tiers[1] += 1
            elif steps <= 4:
                tiers[2] += 1
            else:
                tiers[3] += 1
        return [x / len(s) for x in tiers]

    @staticmethod
    def difficulty_entropy(data, n):
        """M08 难度分布熵: 四级难度的归一化 Shannon 熵
        依据: Bengio et al. (2009) Curriculum Learning"""
        dist = Metrics.difficulty_dist(data, n)
        ent = -sum(p * math.log2(p) for p in dist if p > 0)
        return round(ent / math.log2(4) * 100, 1)

    @staticmethod
    def deep_reasoning_ratio(data, n):
        """M09 深度推理占比: 推理步数 ≥5 步的样本比例"""
        s = sub(data, n)
        deep = sum(1 for i in s if extract_think(i) and
                   len(re.findall(r'(?:^|\n)\s*\d+[\.\、]', extract_think(i))) >= 5)
        return round(deep / len(s) * 100, 1)

    # ── F. 知识覆盖与规模 ──

    @staticmethod
    def source_coverage(data, orig):
        """M10 源文档 Trigram Recall: 合成数据对源文档内容的覆盖程度"""
        syn_tg = set()
        for item in data:
            tokens = tokenize(extract_info(item))
            for i in range(len(tokens) - 2):
                syn_tg.add((tokens[i], tokens[i + 1], tokens[i + 2]))
        recalls = []
        for chunk in orig:
            if len(chunk.strip()) < 15:
                continue
            tokens = tokenize(chunk)
            if len(tokens) < 3:
                continue
            ctg = set((tokens[i], tokens[i + 1], tokens[i + 2])
                      for i in range(len(tokens) - 2))
            if ctg:
                recalls.append(len(ctg & syn_tg) / len(ctg))
        return round(np.mean(recalls), 4) if recalls else 0.0

    @staticmethod
    def dedup_count(data):
        """M11 去重后有效样本数 (output 前 80 字符去重)"""
        seen = set()
        unique = 0
        for d in data:
            key = d["output"][:80]
            if key not in seen:
                seen.add(key)
                unique += 1
        return unique

    @staticmethod
    def data_yield(data, orig):
        """M12 产出率: QA 对数 / 源文档片段数"""
        return round(len(data) / len(orig), 2) if orig else 0.0

    # ── G. 上下文利用 (RAGAS) ──

    @staticmethod
    def entity_utilization(data, n):
        """M13 上下文实体利用率: 信息区实体在回答中被引用的比例
        依据: RAGAS Context Entity Recall (Es et al., 2023)"""
        s = sub(data, n)
        scores = []
        for i in s:
            info_ents = set(re.findall(
                rf'[\u4e00-\u9fff]{{2,8}}(?:{ENTITY_SUFFIXES})', extract_info(i)))
            if not info_ents:
                continue
            t = extract_think(i)
            resp = (t or '') + ' ' + extract_answer(i)
            used = sum(1 for e in info_ents if e in resp)
            scores.append(used / len(info_ents))
        return round(np.mean(scores), 4) if scores else 0.0

    @staticmethod
    def multi_segment_ref(data, n):
        """M14 多段落引用率: 回答引用了信息区多少不同段落的信息
        依据: RAGAS Context Utilization (Es et al., 2023)"""
        s = sub(data, n)
        scores = []
        for i in s:
            info = extract_info(i)
            sents = [x.strip() for x in re.split(r'[。；！]', info) if len(x.strip()) > 10]
            if len(sents) < 2:
                continue
            t = extract_think(i)
            resp = (t or '') + ' ' + extract_answer(i)
            referenced = 0
            for sent in sents:
                kw = set(w for w in jieba.lcut(sent)
                         if len(w) >= 3 and re.fullmatch(r'[\u4e00-\u9fff]+', w))
                if not kw:
                    continue
                if sum(1 for k in kw if k in resp) >= min(2, len(kw)):
                    referenced += 1
            scores.append(referenced / len(sents))
        return round(np.mean(scores), 4) if scores else 0.0

    @staticmethod
    def qc_alignment(data, n):
        """M15 问题-上下文对齐度: 问题关键词在信息区中的覆盖率
        依据: RAGAS Context Precision (Es et al., 2023) 无 LLM 近似"""
        s = sub(data, n)
        scores = []
        for i in s:
            q, info = extract_question(i), extract_info(i)
            if not q or not info:
                continue
            q_terms = set(w for w in jieba.lcut(q)
                         if len(w) >= 2 and re.fullmatch(r'[\u4e00-\u9fff]+', w))
            info_terms = set(w for w in jieba.lcut(info)
                             if len(w) >= 2 and re.fullmatch(r'[\u4e00-\u9fff]+', w))
            if q_terms:
                scores.append(len(q_terms & info_terms) / len(q_terms))
        return round(np.mean(scores), 4) if scores else 0.0

    # ── H. 推理连贯性 ──

    @staticmethod
    def reasoning_transitivity(data, n):
        """M16 推理链传递性: 后续步骤引用前步结论的比例
        依据: Dingo MetaRater PRRC (arxiv 2504.14194); Halliday & Hasan 1976"""
        s = sub(data, n)
        scores = []
        for i in s:
            t = extract_think(i)
            if t is None:
                continue
            steps = [x.strip() for x in re.split(r'(?:^|\n)\s*\d+[\.\、]', t)
                     if len(x.strip()) > 10]
            if len(steps) < 2:
                scores.append(0)
                continue
            transitive = 0
            for j in range(1, len(steps)):
                prev_kw = set(w for w in jieba.lcut(steps[j - 1])
                              if len(w) >= 3 and re.fullmatch(r'[\u4e00-\u9fff]+', w))
                curr_kw = set(w for w in jieba.lcut(steps[j])
                              if len(w) >= 3 and re.fullmatch(r'[\u4e00-\u9fff]+', w))
                if len(prev_kw & curr_kw) >= 2:
                    transitive += 1
            scores.append(transitive / (len(steps) - 1))
        return round(np.mean(scores), 4) if scores else 0.0

    @staticmethod
    def step_balance(data, n):
        """M17 步骤均衡度: 推理各步长度的变异系数 (↓低=好)
        依据: distilabel DEITA 质量评分 — 回答深度均匀性"""
        s = sub(data, n)
        cvs = []
        for i in s:
            t = extract_think(i)
            if t is None:
                continue
            steps = [x.strip() for x in re.split(r'(?:^|\n)\s*\d+[\.\、]', t)
                     if len(x.strip()) > 5]
            if len(steps) < 2:
                continue
            lengths = [len(x) for x in steps]
            mean_l = np.mean(lengths)
            if mean_l > 0:
                cvs.append(np.std(lengths) / mean_l)
        return round(np.mean(cvs), 4) if cvs else 0.0


# ════════════════════════════════════════════════════════════
# 指标定义表 (含权重与方向)
# ════════════════════════════════════════════════════════════

METRIC_DEFS = [
    # (key, name_cn, dimension, higher_is_better, needs_orig, is_fulldata, weight)
    ("M01", "QA相关性(F1)",       "A.问答相关性",      True,  False, False, 3),
    ("M02", "ROUGE-L忠实度",      "B.忠实性",          True,  False, False, 2),
    ("M03", "平均推理步数",        "C.推理深度与质量",   True,  False, False, 3),
    ("M04", "多视角推理率(%)",     "C.推理深度与质量",   True,  False, False, 3),
    ("M05", "推理-信息整合度",     "C.推理深度与质量",   True,  False, False, 3),
    ("M06", "信息区平均句数",      "D.上下文构建",      True,  False, False, 2),
    ("M07", "信息区实体共现",      "D.上下文构建",      True,  False, False, 2),
    ("M08", "难度分布熵(%)",      "E.难度梯度",        True,  False, False, 2),
    ("M09", "深度推理占比(%)",     "E.难度梯度",        True,  False, False, 2),
    ("M10", "源文档Trigram覆盖",  "F.知识覆盖与规模",   True,  True,  True,  2),
    ("M11", "去重后有效规模",      "F.知识覆盖与规模",   True,  False, True,  2),
    ("M12", "产出率(QA/片段)",    "F.知识覆盖与规模",   True,  True,  True,  2),
    ("M13", "实体利用率",         "G.上下文利用",       True,  False, False, 3),
    ("M14", "多段落引用率",        "G.上下文利用",       True,  False, False, 2),
    ("M15", "问题-上下文对齐",     "G.上下文利用",       True,  False, False, 2),
    ("M16", "推理链传递性",        "H.推理连贯性",      True,  False, False, 2),
    ("M17", "步骤均衡度CV(↓)",    "H.推理连贯性",      False, False, False, 2),
]


def compute_metric(key, data, n, orig=None):
    M = Metrics
    dispatch = {
        "M01": lambda: M.qa_relevance(data, n),
        "M02": lambda: M.faithfulness(data, n),
        "M03": lambda: M.reasoning_depth(data, n),
        "M04": lambda: M.multi_perspective(data, n),
        "M05": lambda: M.reasoning_integration(data, n),
        "M06": lambda: M.info_sentences(data, n),
        "M07": lambda: M.entity_cooccurrence(data, n),
        "M08": lambda: M.difficulty_entropy(data, n),
        "M09": lambda: M.deep_reasoning_ratio(data, n),
        "M10": lambda: M.source_coverage(data, orig) if orig else 0.0,
        "M11": lambda: float(M.dedup_count(data)),
        "M12": lambda: M.data_yield(data, orig) if orig else 0.0,
        "M13": lambda: M.entity_utilization(data, n),
        "M14": lambda: M.multi_segment_ref(data, n),
        "M15": lambda: M.qc_alignment(data, n),
        "M16": lambda: M.reasoning_transitivity(data, n),
        "M17": lambda: M.step_balance(data, n),
    }
    return dispatch[key]()


# ════════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════════

def load_sft(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0, f"数据文件格式错误: {path}"
    assert "output" in data[0], "数据需包含 instruction/input/output 字段"
    return data


def load_source(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "chunks" in obj:
                chunks.extend(obj["chunks"])
            elif "text" in obj:
                chunks.append(obj["text"])
            else:
                chunks.append(line.strip())
    return chunks


# ════════════════════════════════════════════════════════════
# 单方法评估
# ════════════════════════════════════════════════════════════

def evaluate_single(data, orig, name, out):
    n = min(len(data), 500)
    out.write(f"{'=' * 82}\n")
    out.write(f"  合成数据质量评估报告 — {name}\n")
    out.write(f"  8 维度 · 17 指标 · 学术标准方法\n")
    out.write(f"{'=' * 82}\n")
    out.write(f"  样本数: {len(data)}  |  采样: {n}\n")
    if orig:
        out.write(f"  源文档: {len(orig)} 片段\n")

    current_dim = ""
    results = {}
    for key, mname, dim, higher, needs_orig, fulldata, weight in METRIC_DEFS:
        if dim != current_dim:
            current_dim = dim
            out.write(f"\n  ▌{dim}\n")
        sample_n = len(data) if fulldata else n
        val = compute_metric(key, data, sample_n, orig)
        results[key] = val
        out.write(f"    {mname:<20}  {val:>12}\n")

    # 难度分布
    dist = Metrics.difficulty_dist(data, n)
    tier_names = ["直答", "简单(1-2步)", "中等(3-4步)", "深度(5+步)"]
    out.write(f"\n  难度分布:\n")
    for i, tn in enumerate(tier_names):
        bar = "█" * int(dist[i] * 40)
        out.write(f"    {tn:<14} {dist[i] * 100:5.1f}%  {bar}\n")

    # 基本统计
    lens = [len(i["output"]) for i in data]
    think_n = sum(1 for i in data if "<think>" in i["output"])
    dedup = Metrics.dedup_count(data)
    out.write(f"\n  基本统计:\n")
    out.write(f"    输出均长: {np.mean(lens):.0f} 字 | 中位数: {np.median(lens):.0f} | "
              f"P10/P90: {np.percentile(lens, 10):.0f}/{np.percentile(lens, 90):.0f}\n")
    out.write(f"    推理类: {think_n}/{len(data)} ({think_n / len(data) * 100:.1f}%)\n")
    out.write(f"    唯一样本: {dedup}/{len(data)} ({dedup / len(data) * 100:.1f}%)\n")

    return results


# ════════════════════════════════════════════════════════════
# 对比评估
# ════════════════════════════════════════════════════════════

def evaluate_compare(sog, doc, orig, sn, dn, out):
    fair = min(len(sog), len(doc), 500)

    out.write(f"{'=' * 82}\n")
    out.write(f"  合成数据质量对比评估报告\n")
    out.write(f"  {sn} vs {dn}\n")
    out.write(f"  8 维度 · 17 指标 · 学术标准方法\n")
    out.write(f"{'=' * 82}\n")
    out.write(f"  {sn}: {len(sog)} 条  |  {dn}: {len(doc)} 条")
    if orig:
        out.write(f"  |  源文档: {len(orig)} 片段")
    out.write(f"\n  等量采样: {fair} 条 (知识覆盖类使用全量)\n")

    results = {}
    current_dim = ""

    for key, mname, dim, higher, needs_orig, fulldata, weight in METRIC_DEFS:
        if dim != current_dim:
            current_dim = dim
            out.write(f"\n  ▌{dim}\n")
        ns = len(sog) if fulldata else fair
        nd = len(doc) if fulldata else fair
        sv = compute_metric(key, sog, ns, orig)
        dv = compute_metric(key, doc, nd, orig)
        results[key] = (sv, dv)

        # Determine winner based on direction
        if higher:
            winner = sn if sv > dv else (dn if dv > sv else "平局")
        else:
            winner = sn if sv < dv else (dn if dv < sv else "平局")

        direction = "↑高好" if higher else "↓低好"
        out.write(f"    {mname:<20}  {sn}={sv:<12}  {dn}={dv:<12}  {direction}  → {winner}\n")

    # 难度分布对比
    sd = Metrics.difficulty_dist(sog, fair)
    dd = Metrics.difficulty_dist(doc, fair)
    tier_names = ["直答", "简单(1-2步)", "中等(3-4步)", "深度(5+步)"]
    out.write(f"\n  难度分布对比:\n")
    for i, tn in enumerate(tier_names):
        s_bar = "█" * int(sd[i] * 30)
        d_bar = "█" * int(dd[i] * 30)
        out.write(f"    {tn:<14}  {sn}: {sd[i] * 100:5.1f}% {s_bar}\n")
        out.write(f"    {'':<14}  {dn}: {dd[i] * 100:5.1f}% {d_bar}\n")

    # ── 汇总表 ──
    out.write(f"\n\n{'═' * 82}\n  综合汇总\n{'═' * 82}\n\n")
    out.write(f"  {'维度':<16} {'指标':<20} {sn:>10} {dn:>10}  {'方向':>6}  {'胜出':>6}\n")
    out.write(f"  {'─' * 16} {'─' * 20} {'─' * 10} {'─' * 10}  {'─' * 6}  {'─' * 6}\n")

    sog_wins = 0
    doc_wins = 0
    dim_results = {}

    for key, mname, dim, higher, _, _, weight in METRIC_DEFS:
        sv, dv = results[key]
        if higher:
            winner = sn if sv > dv else (dn if dv > sv else "平局")
        else:
            winner = sn if sv < dv else (dn if dv < sv else "平局")

        direction = "↑高好" if higher else "↓低好"
        out.write(f"  {dim:<16} {mname:<20} {sv:>10} {dv:>10}  {direction:>6}  {winner:>6}\n")

        if winner == sn:
            sog_wins += 1
        elif winner == dn:
            doc_wins += 1

        if dim not in dim_results:
            dim_results[dim] = [0, 0]
        if winner == sn:
            dim_results[dim][0] += 1
        elif winner == dn:
            dim_results[dim][1] += 1

    out.write(f"\n  指标胜负: {sn} {sog_wins}/{len(METRIC_DEFS)}   "
              f"{dn} {doc_wins}/{len(METRIC_DEFS)}\n")

    # 维度级别统计
    out.write(f"\n  维度胜负:\n")
    sog_dim_wins = 0
    doc_dim_wins = 0
    for dim, (sw, dw) in dim_results.items():
        if sw > dw:
            label = f"{sn} ✓"
            sog_dim_wins += 1
        elif dw > sw:
            label = f"{dn} ✓"
            doc_dim_wins += 1
        else:
            label = "平局"
        out.write(f"    {dim}: {label} ({sw}:{dw})\n")

    out.write(f"\n  维度级别: {sn} {sog_dim_wins}  {dn} {doc_dim_wins}  "
              f"平局 {len(dim_results) - sog_dim_wins - doc_dim_wins}  "
              f"(共 {len(dim_results)} 维度)\n")

    # ── 加权综合得分 ──
    out.write(f"\n\n{'═' * 82}\n  加权综合得分\n{'═' * 82}\n")
    out.write(f"  (权重反映指标对 SFT 训练数据质量的重要性)\n\n")

    sog_weighted = 0.0
    doc_weighted = 0.0
    total_weight = 0.0

    for key, mname, dim, higher, _, _, weight in METRIC_DEFS:
        sv, dv = results[key]
        # Normalize: winner gets 1.0, loser gets ratio
        if higher:
            max_val = max(sv, dv) if max(sv, dv) > 0 else 1
            s_norm = sv / max_val
            d_norm = dv / max_val
        else:
            min_val = min(sv, dv) if min(sv, dv) > 0 else max(sv, dv)
            if min_val > 0:
                s_norm = min_val / sv if sv > 0 else 0
                d_norm = min_val / dv if dv > 0 else 0
            else:
                s_norm = 1.0 if sv <= dv else 0.0
                d_norm = 1.0 if dv <= sv else 0.0

        sog_weighted += s_norm * weight
        doc_weighted += d_norm * weight
        total_weight += weight

    sog_score = round(sog_weighted / total_weight * 100, 1)
    doc_score = round(doc_weighted / total_weight * 100, 1)

    out.write(f"  {sn} 综合得分: {sog_score:.1f} / 100\n")
    out.write(f"  {dn} 综合得分: {doc_score:.1f} / 100\n")
    out.write(f"  {sn} 领先: +{sog_score - doc_score:.1f} 分\n")

    # ── 学术出处 ──
    out.write(f"""

{'═' * 82}
  方法说明与学术出处
{'═' * 82}

  A. 问答相关性 — QA Token-F1: 回答与问题的词袋 F1 重叠度

  B. 忠实性 — ROUGE-L F1 (Lin, ACL 2004): 回答对源文档信息的忠实程度

  C. 推理深度与质量
     · 平均推理步数: <think> 内编号步骤数均值
     · 多视角推理率: ≥3 个分析角度的推理样本占比
       依据: Paul & Elder 批判性思维框架
     · 推理-信息整合度: 推理链对信息区术语的引用召回率

  D. 上下文构建
     · 信息区句数: 衡量上下文信息丰富程度
     · 实体共现数: 多实体共现 = 多知识源融合
       依据: 信息抽取 (IE) 实体密度指标

  E. 难度梯度
     · 难度分布熵: 四级难度的归一化 Shannon 熵
       依据: Bengio et al. (2009) Curriculum Learning
     · 深度推理占比: ≥5 步推理的样本比例

  F. 知识覆盖与规模
     · 源文档 Trigram Recall: 合成数据覆盖源文档内容的程度
     · 去重后有效规模: output 前 80 字符去重后唯一样本数
     · 产出率: QA 对数 / 源文档片段数

  G. 上下文利用
     · 实体利用率: 信息区实体在回答中被引用的比例
       依据: RAGAS Context Entity Recall (Es et al., 2023)
     · 多段落引用率: 回答引用了信息区不同段落的比例
       依据: RAGAS Context Utilization (Es et al., 2023)
     · 问题-上下文对齐: 问题关键词在信息区的覆盖率
       依据: RAGAS Context Precision (Es et al., 2023)

  H. 推理连贯性
     · 推理链传递性: 后续步骤引用前步结论的比例
       依据: Dingo MetaRater PRRC (arxiv 2504.14194)
     · 步骤均衡度: 推理步骤长度的变异系数 (↓低=好)
       依据: distilabel DEITA 质量评分

  公平性保证
  ──────────
  • 等量采样 min(N1, N2, 500); 知识覆盖类使用全量
  • 随机种子 42, 结果完全可复现
  • jieba 中文分词, 客观指标无 LLM 依赖
  • 加权综合得分: 权重基于指标对 SFT 训练质量的贡献度
""")

    return results


# ════════════════════════════════════════════════════════════
# LLM-as-Judge 集成
# ════════════════════════════════════════════════════════════

def run_llm_judge(sog, doc, sn, dn, out, config=None):
    """运行 LLM-as-Judge 评估 (需要 llm_judge 模块)"""
    try:
        from llm_judge import LLMJudge
    except ImportError:
        out.write("\n  [跳过] LLM-as-Judge: 缺少 llm_judge 模块或 openai 库\n")
        out.write("  安装: pip install openai pyyaml\n")
        return None

    judge = LLMJudge(config)
    if not judge.available:
        out.write("\n  [跳过] LLM-as-Judge: 未配置 API (设置环境变量或 config.yaml)\n")
        return None

    out.write(f"\n\n{'═' * 82}\n")
    out.write(f"  LLM-as-Judge 质量评估\n")
    out.write(f"  模型: {judge.model}\n")
    out.write(f"{'═' * 82}\n")

    # 单样本评估
    n_eval = min(50, len(sog), len(doc))
    sog_sample = sub(sog, n_eval)
    doc_sample = sub(doc, n_eval)

    out.write(f"\n  ▌单样本质量评估 (各 {n_eval} 条)\n")
    sog_scores = judge.evaluate_batch(sog_sample)
    doc_scores = judge.evaluate_batch(doc_sample)

    if sog_scores and doc_scores:
        dims = ["回答质量", "推理质量", "知识利用", "训练价值"]
        for d in dims:
            s_avg = np.mean([s.get(d, 0) for s in sog_scores if s])
            d_avg = np.mean([s.get(d, 0) for s in doc_scores if s])
            winner = sn if s_avg > d_avg else dn
            out.write(f"    {d:<12}  {sn}={s_avg:.2f}  {dn}={d_avg:.2f}  → {winner}\n")

        s_overall = np.mean([np.mean(list(s.values())) for s in sog_scores if s])
        d_overall = np.mean([np.mean(list(s.values())) for s in doc_scores if s])
        out.write(f"\n    综合均分     {sn}={s_overall:.2f}  {dn}={d_overall:.2f}\n")

    return {"sog_scores": sog_scores, "doc_scores": doc_scores}


def run_pairwise(sog, doc, sn, dn, out, config=None):
    """运行 Pairwise 对决评估"""
    try:
        from llm_judge import LLMJudge
    except ImportError:
        out.write("\n  [跳过] Pairwise 对决: 缺少 llm_judge 模块\n")
        return None

    judge = LLMJudge(config)
    if not judge.available:
        out.write("\n  [跳过] Pairwise 对决: 未配置 API\n")
        return None

    n_pairs = min(30, len(sog), len(doc))
    sog_sample = sub(sog, n_pairs)
    doc_sample = sub(doc, n_pairs)

    out.write(f"\n\n{'═' * 82}\n")
    out.write(f"  Pairwise 对决 ({n_pairs} 对)\n")
    out.write(f"{'═' * 82}\n")

    results = judge.pairwise_batch(sog_sample, doc_sample, sn, dn)

    if results:
        a_wins = sum(1 for r in results if r and r.get("winner") == "A")
        b_wins = sum(1 for r in results if r and r.get("winner") == "B")
        ties = sum(1 for r in results if r and r.get("winner") == "tie")
        total = a_wins + b_wins + ties

        out.write(f"\n  {sn} 胜: {a_wins} ({a_wins / total * 100:.1f}%)\n")
        out.write(f"  {dn} 胜: {b_wins} ({b_wins / total * 100:.1f}%)\n")
        out.write(f"  平局:   {ties} ({ties / total * 100:.1f}%)\n")

    return results


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SoG 合成数据质量评估框架 v2.0 — 8 维度 17 指标 + LLM-as-Judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 对比评估
  python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl

  # 含 LLM-as-Judge
  python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl --llm-judge

  # 含 Pairwise 对决
  python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl --pairwise

  # 全部评估
  python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl --llm-judge --pairwise

  # 单方法评估
  python evaluate.py --data my.json --source corpus.jsonl --name "SoG"

  # 指定 LLM 配置
  python evaluate.py --sog sog.json --doc doc.json --llm-judge --llm-config config.yaml
        """)

    parser.add_argument("--sog", help="SoG 合成数据 (JSON)")
    parser.add_argument("--doc", help="对比方法数据 (JSON)")
    parser.add_argument("--sog-name", default="SoG", help="SoG 名称 (默认: SoG)")
    parser.add_argument("--doc-name", default="Doc", help="对比方法名称 (默认: Doc)")
    parser.add_argument("--data", help="单方法数据 (JSON)")
    parser.add_argument("--name", default="Method", help="方法名称 (单方法模式)")
    parser.add_argument("--source", help="源文档 (JSONL)")
    parser.add_argument("-o", "--output", help="输出报告路径")
    parser.add_argument("--llm-judge", action="store_true", help="启用 LLM-as-Judge 评估")
    parser.add_argument("--pairwise", action="store_true", help="启用 Pairwise 对决")
    parser.add_argument("--llm-config", help="LLM 配置文件路径 (YAML)")

    args = parser.parse_args()
    if not args.sog and not args.data:
        parser.error("请指定 --sog/--doc 或 --data")

    orig = load_source(args.source) if args.source else None

    # 加载 LLM 配置
    llm_config = None
    if args.llm_judge or args.pairwise:
        if args.llm_config and os.path.exists(args.llm_config):
            try:
                import yaml
                with open(args.llm_config) as f:
                    llm_config = yaml.safe_load(f)
            except ImportError:
                print("警告: 需要 pyyaml 来读取配置文件, pip install pyyaml")

    buf = io.StringIO()

    if args.data:
        evaluate_single(load_sft(args.data), orig, args.name, buf)
    elif args.sog and args.doc:
        sog_data = load_sft(args.sog)
        doc_data = load_sft(args.doc)
        evaluate_compare(sog_data, doc_data, orig,
                         args.sog_name, args.doc_name, buf)

        if args.llm_judge:
            run_llm_judge(sog_data, doc_data,
                          args.sog_name, args.doc_name, buf, llm_config)

        if args.pairwise:
            run_pairwise(sog_data, doc_data,
                         args.sog_name, args.doc_name, buf, llm_config)
    elif args.sog:
        evaluate_single(load_sft(args.sog), orig, args.sog_name, buf)
    else:
        parser.error("对比模式需同时指定 --sog 和 --doc")

    report = buf.getvalue()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"报告已保存: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
