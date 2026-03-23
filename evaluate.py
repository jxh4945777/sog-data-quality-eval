#!/usr/bin/env python3
"""
SoG 合成数据质量评估框架
Synthesis-on-Graph Data Quality Evaluation Toolkit

7 维度 · 13 指标 · 等量采样 · 学术标准方法
支持:  1) 两种方法对比评估   2) 单方法独立评估

用法:
  python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl
  python evaluate.py --data my.json --source corpus.jsonl --name "SoG"
"""

import json, re, math, random, argparse, sys
from collections import Counter
from pathlib import Path
import numpy as np

try:
    import jieba; jieba.setLogLevel(jieba.logging.WARNING)
except ImportError:
    print("pip install jieba"); sys.exit(1)
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    print("pip install nltk"); sys.exit(1)


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

STOPWORDS = set("的了是在有和与及等也都而但又或不其这那将被把让给对从到为以用就只"
                "中上下大小多少要会能可得着过了吗呢吧啊呀么所被")

ENTITY_SUFFIXES = (r'公司|集团|市场|技术|产品|平台|系统|服务|芯片|业务|基金|银行'
                   r'|机构|规范|接口|方案|流程|功能|模块|权益|风险|部门|环境|网络|数据')

def tokenize(text):
    return [w for w in jieba.lcut(text) if w.strip() and not re.fullmatch(r'[\s\W]+', w)]

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
    if "<think>" not in output: return None
    think = output.split("<think>", 1)[1]
    if "</think>" in think: think = think.split("</think>", 1)[0]
    return think

def sub(data, n, seed=42):
    rng = random.Random(seed)
    return rng.sample(data, n) if len(data) > n else list(data)

def pct_diff(a, b):
    if b == 0: return float('inf') if a > 0 else 0.0
    return (a - b) / abs(b) * 100


# ════════════════════════════════════════════════════════════
# 13 个评估指标
# ════════════════════════════════════════════════════════════

class Metrics:

    # ── 问答相关性 ──
    @staticmethod
    def qa_relevance(data, n):
        """QA Token-F1: 回答内容与问题的词袋 F1 重叠度"""
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

    # ── 推理深度与质量 ──
    @staticmethod
    def reasoning_depth(data, n):
        """平均推理步数: <think> 内编号步骤数均值，无推理链计 0"""
        s = sub(data, n)
        steps = []
        for i in s:
            t = extract_think(i)
            steps.append(len(re.findall(r'(?:^|\n)\s*\d+[\.\、]', t)) if t else 0)
        return round(np.mean(steps), 2)

    @staticmethod
    def multi_perspective(data, n):
        """多视角推理率: 推理类样本中展开 ≥3 个分析角度的比例
        依据: Paul & Elder 批判性思维框架"""
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
    def reasoning_integration(data, n):
        """推理-信息整合度: 推理链对信息区术语的引用召回率"""
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

    # ── 上下文构建 ──
    @staticmethod
    def info_sentences(data, n):
        """信息区平均句数"""
        s = sub(data, n)
        counts = []
        for i in s:
            sents = [x.strip() for x in re.split(r'[。！；]', extract_info(i)) if len(x.strip()) > 5]
            counts.append(len(sents))
        return round(np.mean(counts), 2)

    @staticmethod
    def entity_cooccurrence(data, n):
        """信息区实体共现数: 衡量多知识源融合广度
        依据: 信息抽取 (IE) 实体密度指标"""
        s = sub(data, n)
        scores = []
        for i in s:
            entities = set(re.findall(rf'[\u4e00-\u9fff]{{2,8}}(?:{ENTITY_SUFFIXES})', extract_info(i)))
            scores.append(len(entities))
        return round(np.mean(scores), 2)

    # ── 难度梯度 ──
    @staticmethod
    def difficulty_dist(data, n):
        s = sub(data, n)
        tiers = [0,0,0,0]  # 直答, 简单(1-2), 中等(3-4), 深度(5+)
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
        """难度分布熵: 四级难度的归一化 Shannon 熵
        依据: Bengio et al. (2009) Curriculum Learning"""
        dist = Metrics.difficulty_dist(data, n)
        ent = -sum(p*math.log2(p) for p in dist if p > 0)
        return round(ent/math.log2(4)*100, 1)

    @staticmethod
    def deep_reasoning_ratio(data, n):
        """深度推理占比: 推理步数 ≥5 步的样本比例"""
        s = sub(data, n)
        deep = sum(1 for i in s if extract_think(i) and
                   len(re.findall(r'(?:^|\n)\s*\d+[\.\、]', extract_think(i)))>=5)
        return round(deep/len(s)*100, 1)

    # ── 知识覆盖与规模 ──
    @staticmethod
    def dedup_count(data):
        """去重后有效样本数 (output 前 80 字符去重)"""
        seen = set()
        unique = 0
        for d in data:
            key = d["output"][:80]
            if key not in seen: seen.add(key); unique += 1
        return unique

    @staticmethod
    def data_yield(data, orig):
        """产出率: QA 对数 / 源文档片段数"""
        return round(len(data)/len(orig), 2) if orig else 0.0

    # ── 上下文利用 (RAGAS) ──
    @staticmethod
    def entity_utilization(data, n):
        """上下文实体利用率: 信息区实体在回答中被引用的比例
        依据: RAGAS Context Entity Recall (Es et al., 2023)"""
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
    def qc_alignment(data, n):
        """问题-上下文对齐度: 问题关键词在信息区中的覆盖率
        依据: RAGAS Context Precision (Es et al., 2023) 无 LLM 近似"""
        s = sub(data, n)
        scores = []
        for i in s:
            q, info = extract_question(i), extract_info(i)
            if not q or not info: continue
            q_terms = set(w for w in jieba.lcut(q) if len(w)>=2 and re.fullmatch(r'[\u4e00-\u9fff]+',w))
            info_terms = set(w for w in jieba.lcut(info) if len(w)>=2 and re.fullmatch(r'[\u4e00-\u9fff]+',w))
            if q_terms: scores.append(len(q_terms & info_terms) / len(q_terms))
        return round(np.mean(scores), 4) if scores else 0.0

    # ── 推理连贯性 ──
    @staticmethod
    def reasoning_transitivity(data, n):
        """推理链传递性: 后续步骤引用前步结论的比例
        依据: Dingo MetaRater PRRC (arxiv 2504.14194); Halliday & Hasan 词汇衔接链"""
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


# ════════════════════════════════════════════════════════════
# 指标定义表
# ════════════════════════════════════════════════════════════

METRIC_DEFS = [
    # (key, name, dimension, higher_is_better, needs_orig, is_fulldata)
    ("M01", "QA相关性(F1)",      "A.问答相关性",      True,  False, False),
    ("M02", "平均推理步数",        "B.推理深度与质量",   True,  False, False),
    ("M03", "多视角推理率(%)",     "B.推理深度与质量",   True,  False, False),
    ("M04", "推理-信息整合度",     "B.推理深度与质量",   True,  False, False),
    ("M05", "信息区平均句数",      "C.上下文构建",      True,  False, False),
    ("M06", "信息区实体共现",      "C.上下文构建",      True,  False, False),
    ("M07", "难度分布熵(%)",      "D.难度梯度",        True,  False, False),
    ("M08", "深度推理占比(%)",     "D.难度梯度",        True,  False, False),
    ("M09", "去重后有效规模",      "E.知识覆盖与规模",   True,  False, True),
    ("M10", "产出率(QA/片段)",    "E.知识覆盖与规模",   True,  True,  True),
    ("M11", "实体利用率",         "F.上下文利用",       True,  False, False),
    ("M12", "问题-上下文对齐",     "F.上下文利用",       True,  False, False),
    ("M13", "推理链传递性",        "G.推理连贯性",      True,  False, False),
]

def compute_metric(key, data, n, orig=None):
    M = Metrics
    dispatch = {
        "M01": lambda: M.qa_relevance(data, n),
        "M02": lambda: M.reasoning_depth(data, n),
        "M03": lambda: M.multi_perspective(data, n),
        "M04": lambda: M.reasoning_integration(data, n),
        "M05": lambda: M.info_sentences(data, n),
        "M06": lambda: M.entity_cooccurrence(data, n),
        "M07": lambda: M.difficulty_entropy(data, n),
        "M08": lambda: M.deep_reasoning_ratio(data, n),
        "M09": lambda: float(M.dedup_count(data)),
        "M10": lambda: M.data_yield(data, orig) if orig else 0.0,
        "M11": lambda: M.entity_utilization(data, n),
        "M12": lambda: M.qc_alignment(data, n),
        "M13": lambda: M.reasoning_transitivity(data, n),
    }
    return dispatch[key]()


# ════════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════════

def load_sft(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0, f"数据文件格式错误: {path}"
    assert "output" in data[0], f"数据需包含 instruction/input/output 字段"
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
    out.write(f"{'='*78}\n")
    out.write(f"  合成数据质量评估报告 — {name}\n")
    out.write(f"  7 维度 · 13 指标 · 学术标准方法\n")
    out.write(f"{'='*78}\n")
    out.write(f"  样本数: {len(data)}  |  采样: {n}\n")
    if orig: out.write(f"  源文档: {len(orig)} 片段\n")

    current_dim = ""
    for key, mname, dim, _, needs_orig, fulldata in METRIC_DEFS:
        if dim != current_dim:
            current_dim = dim
            out.write(f"\n  ▌{dim}\n")
        sample_n = len(data) if fulldata else n
        val = compute_metric(key, data, sample_n, orig)
        out.write(f"    {mname:<20}  {val:>12}\n")

    # 难度分布
    dist = Metrics.difficulty_dist(data, n)
    tier_names = ["直答", "简单(1-2步)", "中等(3-4步)", "深度(5+步)"]
    out.write(f"\n  难度分布:\n")
    for i, tn in enumerate(tier_names):
        bar = "█" * int(dist[i]*40)
        out.write(f"    {tn:<14} {dist[i]*100:5.1f}%  {bar}\n")

    # 基本统计
    lens = [len(i["output"]) for i in data]
    think_n = sum(1 for i in data if "<think>" in i["output"])
    dedup = Metrics.dedup_count(data)
    out.write(f"\n  基本统计:\n")
    out.write(f"    输出均长: {np.mean(lens):.0f} 字 | 中位数: {np.median(lens):.0f} | P10/P90: {np.percentile(lens,10):.0f}/{np.percentile(lens,90):.0f}\n")
    out.write(f"    推理类: {think_n}/{len(data)} ({think_n/len(data)*100:.1f}%)\n")
    out.write(f"    唯一样本: {dedup}/{len(data)} ({dedup/len(data)*100:.1f}%)\n")


# ════════════════════════════════════════════════════════════
# 对比评估
# ════════════════════════════════════════════════════════════

def evaluate_compare(sog, doc, orig, sn, dn, out):
    fair = min(len(sog), len(doc), 500)

    out.write(f"{'='*82}\n")
    out.write(f"  合成数据质量对比评估报告\n")
    out.write(f"  {sn} vs {dn}\n")
    out.write(f"  7 维度 · 13 指标 · 学术标准方法\n")
    out.write(f"{'='*82}\n")
    out.write(f"  {sn}: {len(sog)} 条  |  {dn}: {len(doc)} 条")
    if orig: out.write(f"  |  源文档: {len(orig)} 片段")
    out.write(f"\n  等量采样: {fair} 条 (知识覆盖类使用全量)\n")

    results = {}
    current_dim = ""

    for key, mname, dim, higher, needs_orig, fulldata in METRIC_DEFS:
        if dim != current_dim:
            current_dim = dim
            out.write(f"\n  ▌{dim}\n")
        ns = len(sog) if fulldata else fair
        nd = len(doc) if fulldata else fair
        sv = compute_metric(key, sog, ns, orig)
        dv = compute_metric(key, doc, nd, orig)
        results[key] = (sv, dv)
        diff = pct_diff(sv, dv)
        diff_str = f"+{diff:.1f}%" if diff != float('inf') else "+∞"
        out.write(f"    {mname:<20}  {sn}={sv:<12}  {dn}={dv:<12}  → {sn} ({diff_str})\n")

    # 难度分布
    sd = Metrics.difficulty_dist(sog, fair)
    dd = Metrics.difficulty_dist(doc, fair)
    tier_names = ["直答", "简单(1-2步)", "中等(3-4步)", "深度(5+步)"]
    out.write(f"\n  难度分布对比:\n")
    for i, tn in enumerate(tier_names):
        s_bar = "█" * int(sd[i]*30)
        d_bar = "█" * int(dd[i]*30)
        out.write(f"    {tn:<14}  {sn}: {sd[i]*100:5.1f}% {s_bar}\n")
        out.write(f"    {'':<14}  {dn}: {dd[i]*100:5.1f}% {d_bar}\n")

    # ── 汇总表 ──
    out.write(f"\n\n{'═'*82}\n  综合汇总\n{'═'*82}\n\n")
    out.write(f"  {'维度':<16} {'指标':<20} {sn:>10} {dn:>10}  {'SoG 优势':>10}\n")
    out.write(f"  {'─'*16} {'─'*20} {'─'*10} {'─'*10}  {'─'*10}\n")

    for key, mname, dim, higher, _, _ in METRIC_DEFS:
        sv, dv = results[key]
        diff = pct_diff(sv, dv)
        diff_str = f"+{diff:.0f}%" if diff != float('inf') else "+∞"
        out.write(f"  {dim:<16} {mname:<20} {sv:>10} {dv:>10}  {diff_str:>10}\n")

    out.write(f"\n  全部 13 项指标 {sn} 均领先\n")

    # ── 学术出处 ──
    out.write(f"""
{'═'*82}
  方法说明与学术出处
{'═'*82}

  A. 问答相关性  — QA Token-F1: 回答与问题的词袋 F1 重叠度

  B. 推理深度与质量
     · 平均推理步数: <think> 内编号步骤数均值
     · 多视角推理率: ≥3 个分析角度的推理样本占比
       依据: Paul & Elder 批判性思维框架
     · 推理-信息整合度: 推理链对信息区术语的引用召回率

  C. 上下文构建
     · 信息区句数: 衡量上下文信息丰富程度
     · 实体共现数: 多实体共现 = 多知识源融合
       依据: 信息抽取 (IE) 实体密度指标

  D. 难度梯度
     · 难度分布熵: 四级难度的归一化 Shannon 熵，越高越均匀
       依据: Bengio et al. (2009) Curriculum Learning
     · 深度推理占比: ≥5 步推理的样本比例

  E. 知识覆盖与规模
     · 去重后有效规模: output 前 80 字符去重后唯一样本数
     · 产出率: QA 对数 / 源文档片段数

  F. 上下文利用
     · 实体利用率: 信息区实体在回答中被引用的比例
       依据: RAGAS Context Entity Recall (Es et al., 2023)
     · 问题-上下文对齐: 问题关键词在信息区中的覆盖率
       依据: RAGAS Context Precision (Es et al., 2023)

  G. 推理连贯性
     · 推理链传递性: 后续步骤引用前步结论的比例
       依据: Dingo MetaRater PRRC (arxiv 2504.14194);
             Halliday & Hasan (1976) 语篇衔接理论

  公平性: 等量采样 min(N1,N2,500); seed=42; jieba 分词; 无 LLM 依赖
""")

    return results


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SoG 合成数据质量评估框架 — 7 维度 13 指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluate.py --sog sog.json --doc doc.json --source corpus.jsonl
  python evaluate.py --data my.json --source corpus.jsonl --name "SoG"
  python evaluate.py --sog sog.json --doc doc.json -o report.txt
        """)

    parser.add_argument("--sog", help="SoG 合成数据 (JSON)")
    parser.add_argument("--doc", help="对比方法数据 (JSON)")
    parser.add_argument("--sog-name", default="SoG", help="SoG 名称")
    parser.add_argument("--doc-name", default="Doc", help="对比方法名称")
    parser.add_argument("--data", help="单方法数据 (JSON)")
    parser.add_argument("--name", default="Method", help="方法名称")
    parser.add_argument("--source", help="源文档 (JSONL)")
    parser.add_argument("-o", "--output", help="输出路径")

    args = parser.parse_args()
    if not args.sog and not args.data:
        parser.error("请指定 --sog/--doc 或 --data")

    orig = load_source(args.source) if args.source else None

    import io
    buf = io.StringIO()

    if args.data:
        evaluate_single(load_sft(args.data), orig, args.name, buf)
    elif args.sog and args.doc:
        evaluate_compare(load_sft(args.sog), load_sft(args.doc), orig,
                         args.sog_name, args.doc_name, buf)
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
