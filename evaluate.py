#!/usr/bin/env python3
"""
SoG 合成数据质量评估框架
Synthesis-on-Graph Data Quality Evaluation Toolkit

10 维度 · 24 指标 · 等量采样 · 学术标准方法
支持:  1) 两种方法对比评估   2) 单方法独立评估

指标来源: Distinct-N / Self-BLEU / ROUGE-L / RAGAS / Dingo / Data-Juicer / distilabel

用法:
  # 对比两种方法
  python evaluate.py --sog sog_data.json --doc doc_data.json --source corpus.jsonl

  # 单独评估一种方法
  python evaluate.py --data my_data.json --source corpus.jsonl --name "SoG"

  # 指定输出路径
  python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl -o report.txt
"""

import json, re, math, gzip, random, argparse, sys
from collections import Counter
from pathlib import Path
import numpy as np

try:
    import jieba
    jieba.setLogLevel(jieba.logging.WARNING)
except ImportError:
    print("Error: 请安装 jieba: pip install jieba"); sys.exit(1)

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    print("Error: 请安装 nltk: pip install nltk"); sys.exit(1)


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

STOPWORDS = set("的了是在有和与及等也都而但又或不其这那将被把让给对从到为以用就只"
                "中上下大小多少要会能可得着过了吗呢吧啊呀么所被")

def tokenize(text):
    return [w for w in jieba.lcut(text) if w.strip() and not re.fullmatch(r'[\s\W]+', w)]

def tokenize_content(text):
    text = re.sub(r'</?(?:think|answer)>', '', text)
    text = re.sub(r'(?:^|\n)\s*\d+[\.\、\)]\s*', ' ', text)
    text = re.sub(r'\*\*', '', text)
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) >= 2]

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
    if m == 0 or n == 0: return 0
    prev = [0]*(n+1); curr = [0]*(n+1)
    for i in range(1, m+1):
        for j in range(1, n+1):
            curr[j] = prev[j-1]+1 if x[i-1]==y[j-1] else max(curr[j-1], prev[j])
        prev, curr = curr, [0]*(n+1)
    return prev[n]

def rouge_l_f1(hyp, ref):
    if not hyp or not ref: return 0.0
    l = lcs_len(hyp, ref)
    if l == 0: return 0.0
    p, r = l/len(hyp), l/len(ref)
    return 2*p*r/(p+r)

def pct_diff(a, b):
    """计算百分比差异 (a 相对 b 的变化)"""
    if b == 0: return float('inf') if a > 0 else 0.0
    return (a - b) / abs(b) * 100


# ════════════════════════════════════════════════════════════
# 24 个评估指标
# ════════════════════════════════════════════════════════════

ENTITY_SUFFIXES = (r'公司|集团|市场|技术|产品|平台|系统|服务|芯片|业务|基金|银行'
                   r'|机构|规范|接口|方案|流程|功能|模块|权益|风险|部门|环境|网络|数据')

class Metrics:
    """所有评估指标的计算"""

    # ── A. 多样性 ──
    @staticmethod
    def distinct2(data, n):
        s = sub(data, n)
        tok = [tokenize_content(i["output"]) for i in s]
        bg = []
        for t in tok:
            bg.extend((t[i],t[i+1]) for i in range(len(t)-1))
        return round(len(set(bg))/len(bg), 4) if bg else 0.0

    @staticmethod
    def self_bleu(data, n):
        s = sub(data, n)
        tok = [tokenize_content(i["output"]) for i in s]
        sm = SmoothingFunction().method1
        rng = random.Random(42)
        rn = min(100, len(tok)-1)
        scores = []
        for i, hyp in enumerate(tok):
            if len(hyp) < 4: continue
            idx = [j for j in range(len(tok)) if j!=i]
            ri = rng.sample(idx, min(rn, len(idx)))
            try:
                scores.append(sentence_bleu([tok[j] for j in ri], hyp,
                              weights=(.25,.25,.25,.25), smoothing_function=sm))
            except: continue
        return round(np.mean(scores), 4) if scores else 0.0

    # ── B. 忠实性 ──
    @staticmethod
    def faithfulness(data, n):
        s = sub(data, n)
        return round(np.mean([rouge_l_f1(tokenize(extract_answer(i))[:200],
                                         tokenize(extract_info(i))[:200]) for i in s]), 4)

    @staticmethod
    def qa_relevance(data, n):
        s = sub(data, n)
        scores = []
        for i in s:
            q = extract_question(i)
            if not q: continue
            qc, ac = Counter(tokenize(q)), Counter(tokenize(extract_answer(i)))
            c = sum((qc & ac).values())
            if c == 0: scores.append(0.0); continue
            p, r = c/sum(ac.values()), c/sum(qc.values())
            scores.append(2*p*r/(p+r))
        return round(np.mean(scores), 4)

    # ── C. 信息密度 ──
    @staticmethod
    def compression(data, n):
        s = sub(data, n)
        raw = "\n".join(i["output"] for i in s).encode("utf-8")
        return round(len(raw)/len(gzip.compress(raw)), 4)

    @staticmethod
    def content_density(data, n):
        s = sub(data, n)
        scores = []
        for i in s:
            o = i["output"]
            ct = tokenize_content(o)
            if len(o) < 10: continue
            scores.append(len(set(ct))/len(o)*100)
        return round(np.mean(scores), 4)

    # ── D. 推理深度 ──
    @staticmethod
    def reasoning_depth(data, n):
        s = sub(data, n)
        steps = []
        for i in s:
            t = extract_think(i)
            steps.append(len(re.findall(r'(?:^|\n)\s*\d+[\.\、]', t)) if t else 0)
        return round(np.mean(steps), 2)

    @staticmethod
    def conclusion_rate(data, n):
        s = sub(data, n)
        reasoning = [i for i in s if extract_think(i) is not None]
        if not reasoning: return 0.0
        concluded = sum(1 for i in reasoning
                        if re.search(r'答案是|因此[，。]|综上|总之|结论[是为]', i["output"]))
        return round(concluded/len(reasoning)*100, 1)

    # ── E. 推理质量 ──
    @staticmethod
    def multi_perspective(data, n):
        s = sub(data, n)
        reasoning = [i for i in s if extract_think(i) is not None]
        if not reasoning: return 0.0
        count = 0
        for i in reasoning:
            t = extract_think(i)
            steps = len(re.findall(r'(?:^|\n)\s*\d+[\.\、]', t))
            has_multi = bool(re.search(r'一方面.*另一方面|不仅.*还|既.*又', t, re.DOTALL))
            if steps >= 3 or has_multi: count += 1
        return round(count/len(reasoning)*100, 1)

    @staticmethod
    def logic_density(data, n):
        s = sub(data, n)
        scores = []
        for i in s:
            t = extract_think(i)
            if t is None or len(t) < 20: continue
            causal = len(re.findall(r'因此|所以|导致|由于|因为|从而|使得|意味着', t))
            contrast = len(re.findall(r'然而|但是|不过|尽管|虽然|相反|反而', t))
            progress = len(re.findall(r'此外|而且|进而|进一步|同时|不仅|更', t))
            scores.append((causal + contrast + progress) / len(t) * 100)
        return round(np.mean(scores), 3) if scores else 0.0

    @staticmethod
    def reasoning_integration(data, n):
        s = sub(data, n)
        scores = []
        for i in s:
            t = extract_think(i)
            if t is None: continue
            info_terms = set(w for w in jieba.lcut(extract_info(i))
                             if len(w)>=3 and re.fullmatch(r'[\u4e00-\u9fff]+',w))
            think_terms = set(w for w in jieba.lcut(t)
                              if len(w)>=3 and re.fullmatch(r'[\u4e00-\u9fff]+',w))
            if info_terms:
                scores.append(len(info_terms & think_terms) / len(info_terms))
        return round(np.mean(scores), 4) if scores else 0.0

    # ── F. 上下文构建 ──
    @staticmethod
    def info_sentences(data, n):
        s = sub(data, n)
        counts = []
        for i in s:
            sents = [x.strip() for x in re.split(r'[。！；]', extract_info(i)) if len(x.strip()) > 5]
            counts.append(len(sents))
        return round(np.mean(counts), 2)

    @staticmethod
    def entity_cooccurrence(data, n):
        s = sub(data, n)
        scores = []
        for i in s:
            entities = set(re.findall(rf'[\u4e00-\u9fff]{{2,8}}(?:{ENTITY_SUFFIXES})', extract_info(i)))
            scores.append(len(entities))
        return round(np.mean(scores), 2)

    # ── G. 难度梯度 ──
    @staticmethod
    def difficulty_dist(data, n):
        s = sub(data, n)
        tiers = [0,0,0,0]
        for i in s:
            t = extract_think(i)
            if t is None: tiers[0]+=1; continue
            steps = len(re.findall(r'(?:^|\n)\s*\d+[\.\、]', t))
            if steps<=2: tiers[1]+=1
            elif steps<=4: tiers[2]+=1
            else: tiers[3]+=1
        return [x/len(s) for x in tiers]

    @staticmethod
    def difficulty_entropy(data, n):
        dist = Metrics.difficulty_dist(data, n)
        ent = -sum(p*math.log2(p) for p in dist if p > 0)
        return round(ent/math.log2(4)*100, 1)

    @staticmethod
    def deep_reasoning_ratio(data, n):
        s = sub(data, n)
        deep = sum(1 for i in s if extract_think(i) and
                   len(re.findall(r'(?:^|\n)\s*\d+[\.\、]', extract_think(i)))>=5)
        return round(deep/len(s)*100, 1)

    # ── H. 知识覆盖 ──
    @staticmethod
    def source_coverage(data, orig):
        syn_tg = set()
        for item in data:
            tokens = tokenize(extract_info(item))
            for i in range(len(tokens)-2):
                syn_tg.add((tokens[i], tokens[i+1], tokens[i+2]))
        recalls = []
        for chunk in orig:
            if len(chunk.strip()) < 15: continue
            tokens = tokenize(chunk)
            if len(tokens) < 3: continue
            ctg = set((tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2))
            if ctg: recalls.append(len(ctg & syn_tg) / len(ctg))
        return round(np.mean(recalls), 4) if recalls else 0.0

    @staticmethod
    def dedup_count(data):
        seen = set()
        unique = 0
        for d in data:
            key = d["output"][:80]
            if key not in seen: seen.add(key); unique += 1
        return unique

    @staticmethod
    def data_yield(data, orig):
        return round(len(data)/len(orig), 2) if orig else 0.0

    # ── I. 上下文利用 (RAGAS) ──
    @staticmethod
    def entity_utilization(data, n):
        s = sub(data, n)
        scores = []
        for i in s:
            info_ents = set(re.findall(rf'[\u4e00-\u9fff]{{2,8}}(?:{ENTITY_SUFFIXES})', extract_info(i)))
            if not info_ents: continue
            t = extract_think(i)
            resp = (t or '') + ' ' + extract_answer(i)
            used = sum(1 for e in info_ents if e in resp)
            scores.append(used / len(info_ents))
        return round(np.mean(scores), 4) if scores else 0.0

    @staticmethod
    def multi_segment_ref(data, n):
        s = sub(data, n)
        scores = []
        for i in s:
            sents = [x.strip() for x in re.split(r'[。；！]', extract_info(i)) if len(x.strip()) > 10]
            if len(sents) < 2: continue
            t = extract_think(i)
            resp = (t or '') + ' ' + extract_answer(i)
            referenced = 0
            for sent in sents:
                kw = set(w for w in jieba.lcut(sent) if len(w)>=3 and re.fullmatch(r'[\u4e00-\u9fff]+',w))
                if kw and sum(1 for k in kw if k in resp)>=min(2,len(kw)): referenced+=1
            scores.append(referenced / len(sents))
        return round(np.mean(scores), 4) if scores else 0.0

    @staticmethod
    def qc_alignment(data, n):
        s = sub(data, n)
        scores = []
        for i in s:
            q, info = extract_question(i), extract_info(i)
            if not q or not info: continue
            q_terms = set(w for w in jieba.lcut(q) if len(w)>=2 and re.fullmatch(r'[\u4e00-\u9fff]+',w))
            info_terms = set(w for w in jieba.lcut(info) if len(w)>=2 and re.fullmatch(r'[\u4e00-\u9fff]+',w))
            if q_terms: scores.append(len(q_terms & info_terms) / len(q_terms))
        return round(np.mean(scores), 4) if scores else 0.0

    # ── J. 推理连贯性 (Dingo/Data-Juicer) ──
    @staticmethod
    def reasoning_transitivity(data, n):
        s = sub(data, n)
        scores = []
        for i in s:
            t = extract_think(i)
            if t is None: continue
            steps = [x.strip() for x in re.split(r'(?:^|\n)\s*\d+[\.\、]', t) if len(x.strip()) > 10]
            if len(steps) < 2: scores.append(0); continue
            transitive = 0
            for j in range(1, len(steps)):
                prev_kw = set(w for w in jieba.lcut(steps[j-1]) if len(w)>=3 and re.fullmatch(r'[\u4e00-\u9fff]+',w))
                curr_kw = set(w for w in jieba.lcut(steps[j]) if len(w)>=3 and re.fullmatch(r'[\u4e00-\u9fff]+',w))
                if len(prev_kw & curr_kw) >= 2: transitive += 1
            scores.append(transitive / (len(steps)-1))
        return round(np.mean(scores), 4) if scores else 0.0

    @staticmethod
    def intra_repetition(data, n):
        s = sub(data, n)
        scores = []
        for i in s:
            tokens = jieba.lcut(i["output"])
            if len(tokens) < 20: continue
            ngrams = [tuple(tokens[j:j+4]) for j in range(len(tokens)-3)]
            counts = Counter(ngrams)
            repeated = sum(c-1 for c in counts.values() if c > 1)
            scores.append(repeated / max(len(ngrams),1))
        return round(np.mean(scores), 4) if scores else 0.0

    @staticmethod
    def step_balance(data, n):
        s = sub(data, n)
        cvs = []
        for i in s:
            t = extract_think(i)
            if t is None: continue
            steps = [x.strip() for x in re.split(r'(?:^|\n)\s*\d+[\.\、]', t) if len(x.strip()) > 5]
            if len(steps) < 2: continue
            lengths = [len(x) for x in steps]
            mean_l = np.mean(lengths)
            if mean_l > 0: cvs.append(np.std(lengths) / mean_l)
        return round(np.mean(cvs), 4) if cvs else 0.0


# ════════════════════════════════════════════════════════════
# 指标定义表
# ════════════════════════════════════════════════════════════

METRIC_DEFS = [
    # key, name, dimension, higher_is_better, needs_orig, is_fulldata
    ("A1", "Distinct-2", "A.多样性", True, False, False),
    ("A2", "Self-BLEU-4", "A.多样性", False, False, False),
    ("B1", "ROUGE-L忠实度", "B.忠实性", True, False, False),
    ("B2", "QA Token-F1", "B.忠实性", True, False, False),
    ("C1", "Compression", "C.信息密度", False, False, False),
    ("C2", "内容词密度", "C.信息密度", True, False, False),
    ("D1", "推理步数", "D.推理深度", True, False, False),
    ("D2", "结论完整率", "D.推理深度", True, False, False),
    ("E1", "多视角推理", "E.推理质量", True, False, False),
    ("E2", "逻辑连接密度", "E.推理质量", True, False, False),
    ("E3", "推理整合度", "E.推理质量", True, False, False),
    ("F1", "信息区句数", "F.上下文构建", True, False, False),
    ("F2", "实体共现数", "F.上下文构建", True, False, False),
    ("G1", "难度分布熵", "G.难度梯度", True, False, False),
    ("G2", "深度推理占比", "G.难度梯度", True, False, False),
    ("H1", "源文档覆盖", "H.知识覆盖", True, True, True),
    ("H2", "去重后规模", "H.知识覆盖", True, False, True),
    ("H3", "产出率", "H.知识覆盖", True, True, True),
    ("I1", "实体利用率", "I.上下文利用", True, False, False),
    ("I2", "多段落引用", "I.上下文利用", True, False, False),
    ("I3", "问题上下文对齐", "I.上下文利用", True, False, False),
    ("J1", "推理传递性", "J.推理连贯性", True, False, False),
    ("J2", "样本内重复", "J.推理连贯性", False, False, False),
    ("J3", "步骤均衡度", "J.推理连贯性", False, False, False),
]

def compute_metric(key, data, n, orig=None):
    """计算单个指标"""
    M = Metrics
    dispatch = {
        "A1": lambda: M.distinct2(data, n),
        "A2": lambda: M.self_bleu(data, n),
        "B1": lambda: M.faithfulness(data, n),
        "B2": lambda: M.qa_relevance(data, n),
        "C1": lambda: M.compression(data, n),
        "C2": lambda: M.content_density(data, n),
        "D1": lambda: M.reasoning_depth(data, n),
        "D2": lambda: M.conclusion_rate(data, n),
        "E1": lambda: M.multi_perspective(data, n),
        "E2": lambda: M.logic_density(data, n),
        "E3": lambda: M.reasoning_integration(data, n),
        "F1": lambda: M.info_sentences(data, n),
        "F2": lambda: M.entity_cooccurrence(data, n),
        "G1": lambda: M.difficulty_entropy(data, n),
        "G2": lambda: M.deep_reasoning_ratio(data, n),
        "H1": lambda: M.source_coverage(data, orig) if orig else 0.0,
        "H2": lambda: float(M.dedup_count(data)),
        "H3": lambda: M.data_yield(data, orig) if orig else 0.0,
        "I1": lambda: M.entity_utilization(data, n),
        "I2": lambda: M.multi_segment_ref(data, n),
        "I3": lambda: M.qc_alignment(data, n),
        "J1": lambda: M.reasoning_transitivity(data, n),
        "J2": lambda: M.intra_repetition(data, n),
        "J3": lambda: M.step_balance(data, n),
    }
    return dispatch[key]()


# ════════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════════

def load_sft(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0, f"数据文件格式错误: {path}"
    assert "output" in data[0], f"数据格式需包含 instruction/input/output 字段"
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
    out.write(f"{'='*80}\n")
    out.write(f"  合成数据质量评估报告 — {name}\n")
    out.write(f"  样本数: {len(data)}  |  等量采样: {n}\n")
    if orig:
        out.write(f"  源文档: {len(orig)} 片段\n")
    out.write(f"{'='*80}\n\n")

    results = {}
    current_dim = ""
    for key, mname, dim, higher, needs_orig, fulldata in METRIC_DEFS:
        if dim != current_dim:
            current_dim = dim
            out.write(f"\n  ▌{dim}\n")
        sample_n = len(data) if fulldata else n
        val = compute_metric(key, data, sample_n, orig)
        results[key] = val
        direction = "↑高好" if higher else "↓低好"
        out.write(f"    {mname:<16}  {val:>12}  ({direction})\n")

    # 难度分布明细
    dist = Metrics.difficulty_dist(data, n)
    tier_names = ["直答", "简单(1-2步)", "中等(3-4步)", "深度(5+步)"]
    out.write(f"\n  难度分布明细:\n")
    for i, tn in enumerate(tier_names):
        bar = "█" * int(dist[i]*40)
        out.write(f"    {tn:<14} {dist[i]*100:5.1f}%  {bar}\n")

    # 基本统计
    out.write(f"\n  基本统计:\n")
    lens = [len(i["output"]) for i in data]
    out.write(f"    输出均长: {np.mean(lens):.0f} 字  中位数: {np.median(lens):.0f}  P10/P90: {np.percentile(lens,10):.0f}/{np.percentile(lens,90):.0f}\n")
    think_n = sum(1 for i in data if "<think>" in i["output"])
    out.write(f"    推理类占比: {think_n}/{len(data)} ({think_n/len(data)*100:.1f}%)\n")
    dedup = Metrics.dedup_count(data)
    out.write(f"    去重后: {dedup}/{len(data)} ({dedup/len(data)*100:.1f}% 唯一)\n")

    return results


# ════════════════════════════════════════════════════════════
# 对比评估
# ════════════════════════════════════════════════════════════

def evaluate_compare(sog, doc, orig, sog_name, doc_name, out):
    fair = min(len(sog), len(doc), 500)
    out.write(f"{'='*84}\n")
    out.write(f"  合成数据质量对比评估报告\n")
    out.write(f"  {sog_name} vs {doc_name}\n")
    out.write(f"  10 维度 · 24 指标 · 学术标准方法\n")
    out.write(f"{'='*84}\n")
    out.write(f"  {sog_name}: {len(sog)} 条  |  {doc_name}: {len(doc)} 条")
    if orig:
        out.write(f"  |  源文档: {len(orig)} 片段")
    out.write(f"\n  等量采样: {fair} 条 (知识覆盖使用全量)\n")

    results_s = {}
    results_d = {}
    current_dim = ""

    for key, mname, dim, higher, needs_orig, fulldata in METRIC_DEFS:
        if dim != current_dim:
            current_dim = dim
            out.write(f"\n  ▌{dim}\n")
        sample_n = len(sog) if fulldata else fair
        sample_n_d = len(doc) if fulldata else fair

        sv = compute_metric(key, sog, sample_n, orig)
        dv = compute_metric(key, doc, sample_n_d, orig)
        results_s[key] = sv
        results_d[key] = dv

        winner_is_sog = (sv > dv) == higher
        winner = sog_name if winner_is_sog else doc_name
        direction = "↑" if higher else "↓"
        diff = pct_diff(sv, dv) if higher else pct_diff(dv, sv)
        diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
        out.write(f"    {mname:<16}  {sog_name}={sv:<12}  {doc_name}={dv:<12}  {direction} → {winner} ({diff_str})\n")

    # 难度分布明细
    sd = Metrics.difficulty_dist(sog, fair)
    dd = Metrics.difficulty_dist(doc, fair)
    tier_names = ["直答", "简单(1-2步)", "中等(3-4步)", "深度(5+步)"]
    out.write(f"\n  难度分布对比:\n")
    for i, tn in enumerate(tier_names):
        s_bar = "█" * int(sd[i]*30)
        d_bar = "█" * int(dd[i]*30)
        out.write(f"    {tn:<14}  {sog_name}: {sd[i]*100:5.1f}% {s_bar}\n")
        out.write(f"    {'':<14}  {doc_name}: {dd[i]*100:5.1f}% {d_bar}\n")

    # ── 汇总表 ──
    out.write(f"\n\n{'═'*84}\n  综合汇总\n{'═'*84}\n\n")
    out.write(f"  {'维度':<12} {'指标':<16} {sog_name:>10} {doc_name:>10}  {'差异':>8}  {'方向':>4}  {'胜出':>4}\n")
    out.write(f"  {'─'*12} {'─'*16} {'─'*10} {'─'*10}  {'─'*8}  {'─'*4}  {'─'*4}\n")

    sog_w = doc_w = 0
    dims = {}
    for key, mname, dim, higher, _, _ in METRIC_DEFS:
        sv, dv = results_s[key], results_d[key]
        w = (sv > dv) == higher
        if w: sog_w += 1
        else: doc_w += 1
        winner = sog_name if w else doc_name
        diff = pct_diff(sv, dv) if higher else pct_diff(dv, sv)
        diff_str = f"+{diff:.0f}%" if diff > 0 else f"{diff:.0f}%"
        direction = "↑" if higher else "↓"
        out.write(f"  {dim:<12} {mname:<16} {sv:>10} {dv:>10}  {diff_str:>8}  {direction:>4}  {winner:>4}\n")
        dims.setdefault(dim, []).append(sog_name if w else doc_name)

    out.write(f"\n  指标胜负: {sog_name} {sog_w}/24   {doc_name} {doc_w}/24\n")

    # 维度级别
    out.write(f"\n  维度胜负:\n")
    sog_dim = doc_dim = tie_dim = 0
    for dim, ws in dims.items():
        sc, dc = ws.count(sog_name), ws.count(doc_name)
        if sc > dc: verdict = f"{sog_name} ✓"; sog_dim += 1
        elif dc > sc: verdict = f"{doc_name} ✓"; doc_dim += 1
        else: verdict = "平局"; tie_dim += 1
        out.write(f"    {dim}: {verdict} ({sc}:{dc})\n")
    out.write(f"\n  维度: {sog_name} {sog_dim}  {doc_name} {doc_dim}  平局 {tie_dim}\n")

    # ── 学术出处 ──
    out.write(f"""
{'═'*84}
  方法说明与学术出处
{'═'*84}

  A. 多样性 — Distinct-2 (Li et al., NAACL 2016), Self-BLEU-4 (Zhu et al., SIGIR 2018)
  B. 忠实性 — ROUGE-L F1 (Lin, ACL 2004); QA Token-F1
  C. 信息密度 — gzip Compression Ratio (SemDeDup/D4); 内容词密度
  D. 推理深度 — 平均推理步数; 结论完整率
  E. 推理质量 — 多视角推理率 (Paul & Elder 批判性思维); 逻辑连接密度 (Halliday
     & Hasan 1976 语篇衔接理论; Schiffrin 1987); 推理-信息整合度
  F. 上下文构建 — 信息区句数; 实体共现数 (IE 实体密度指标)
  G. 难度梯度 — 难度分布熵 (Bengio et al. 2009 Curriculum Learning); 深度推理占比
  H. 知识覆盖 — Trigram Recall (全量); 去重后规模; 产出率
  I. 上下文利用 — 实体利用率 (RAGAS Context Entity Recall, Es et al. 2023);
     多段落引用率 (RAGAS Context Utilization); 问题-上下文对齐 (RAGAS Context Precision)
  J. 推理连贯性 — 推理链传递性 (Dingo PRRC, arxiv 2504.14194);
     样本内重复率 (Data-Juicer word_repetition_filter);
     步骤均衡度 (distilabel DEITA, Liu et al. ICLR 2024)

  公平性: 双方等量采样 min(N1,N2,500); 随机种子 42; jieba 分词; 无 LLM 依赖
""")

    return results_s, results_d


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SoG 合成数据质量评估框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 对比评估
  python evaluate.py --sog sog_data.json --doc doc_data.json --source corpus.jsonl

  # 单独评估
  python evaluate.py --data my_data.json --source corpus.jsonl --name "My Method"

  # 自定义命名
  python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl \\
                     --sog-name "SoG" --doc-name "Baseline" -o report.txt
        """)

    # 对比模式
    parser.add_argument("--sog", help="SoG 合成数据路径 (JSON)")
    parser.add_argument("--doc", help="Doc 合成数据路径 (JSON)")
    parser.add_argument("--sog-name", default="SoG", help="SoG 方法名称 (默认: SoG)")
    parser.add_argument("--doc-name", default="Doc", help="Doc 方法名称 (默认: Doc)")

    # 单独模式
    parser.add_argument("--data", help="单方法数据路径 (JSON)")
    parser.add_argument("--name", default="Method", help="方法名称 (默认: Method)")

    # 通用
    parser.add_argument("--source", help="源文档路径 (JSONL，含 chunks 字段)")
    parser.add_argument("-o", "--output", help="输出报告路径 (默认: 标准输出)")

    args = parser.parse_args()

    if not args.sog and not args.data:
        parser.error("请指定 --sog/--doc (对比模式) 或 --data (单独模式)")

    orig = load_source(args.source) if args.source else None

    import io
    buf = io.StringIO()

    if args.data:
        # 单独模式
        data = load_sft(args.data)
        evaluate_single(data, orig, args.name, buf)
    elif args.sog and args.doc:
        # 对比模式
        sog = load_sft(args.sog)
        doc = load_sft(args.doc)
        evaluate_compare(sog, doc, orig, args.sog_name, args.doc_name, buf)
    elif args.sog:
        data = load_sft(args.sog)
        evaluate_single(data, orig, args.sog_name, buf)
    else:
        parser.error("对比模式需同时指定 --sog 和 --doc")

    report = buf.getvalue()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"报告已保存到: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
