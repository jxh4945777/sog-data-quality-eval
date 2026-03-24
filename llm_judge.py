#!/usr/bin/env python3
"""
LLM-as-Judge 评估模块
支持 OpenAI 兼容 API (云端/本地部署)

配置方式 (按优先级):
  1. 传入 config dict
  2. config.yaml 文件
  3. 环境变量: LLM_API_BASE, LLM_API_KEY, LLM_MODEL
  4. OPENAI_BASE_URL, OPENAI_API_KEY (OpenAI 标准变量)
"""

import json, os, re, sys, time, random

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ════════════════════════════════════════════════════════════
# 评估 Prompt 模板
# ════════════════════════════════════════════════════════════

SINGLE_EVAL_PROMPT = """你是一位专业的合成训练数据质量评估专家。请评估以下 QA 训练样本的质量。

【信息区】
{info}

【问题】
{question}

【回答】
{answer}

请从以下 4 个维度对该样本打分 (1-5 分):

1. **回答质量**: 回答是否完整、准确，是否具体引用了信息区中的数据和细节？
   - 5分: 全面准确，引用具体数据，结构清晰
   - 3分: 基本回答了问题，但缺乏细节
   - 1分: 回答模糊、不完整或与问题无关

2. **推理质量**: 回答的推理过程是否逻辑严密、层层递进，是否从多个角度分析？
   - 5分: 多步推理，逻辑链完整，多角度分析
   - 3分: 有基本推理但较浅
   - 1分: 无推理过程或逻辑混乱

3. **知识利用**: 回答是否充分利用了信息区提供的知识，实体和关键术语的利用程度如何？
   - 5分: 充分利用多个信息点，综合多段内容
   - 3分: 利用了部分信息
   - 1分: 几乎未利用提供的信息

4. **训练价值**: 作为 SFT 训练数据，该样本对模型学习复杂推理和知识整合能力的教学价值如何？
   - 5分: 展示了高质量推理模式，有很强的教学价值
   - 3分: 有一定训练价值
   - 1分: 训练价值低，过于简单或有错误

请以 JSON 格式输出评分结果:
{{"回答质量": <1-5>, "推理质量": <1-5>, "知识利用": <1-5>, "训练价值": <1-5>}}

只输出 JSON，不要其他内容。"""


PAIRWISE_PROMPT = """你是一位专业的合成训练数据质量评估专家。请比较以下两个 QA 训练样本的质量。

【样本 A】
信息区: {info_a}
问题: {question_a}
回答: {answer_a}

【样本 B】
信息区: {info_b}
问题: {question_b}
回答: {answer_b}

请从以下维度综合评判哪个样本质量更高:
1. 回答的完整性、准确性和深度
2. 推理过程的逻辑性和多角度分析能力
3. 对上下文信息的利用程度
4. 作为 SFT 训练数据的教学价值

请以 JSON 格式输出:
{{"winner": "A" 或 "B" 或 "tie", "score_a": <1-5>, "score_b": <1-5>, "reason": "简要说明理由"}}

只输出 JSON，不要其他内容。"""


# ════════════════════════════════════════════════════════════
# LLM Judge 类
# ════════════════════════════════════════════════════════════

class LLMJudge:
    def __init__(self, config=None):
        self.client = None
        self.model = None
        self.available = False

        if not HAS_OPENAI:
            return

        config = config or {}

        # 从配置或环境变量获取设置
        api_base = (config.get("api_base")
                    or os.environ.get("LLM_API_BASE")
                    or os.environ.get("OPENAI_BASE_URL")
                    or "https://api.openai.com/v1")

        api_key = (config.get("api_key")
                   or os.environ.get("LLM_API_KEY")
                   or os.environ.get("OPENAI_API_KEY")
                   or "")

        self.model = (config.get("model")
                      or os.environ.get("LLM_MODEL")
                      or os.environ.get("OPENAI_MODEL")
                      or "gpt-4o-mini")

        self.temperature = config.get("temperature", 0.1)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)

        if not api_key:
            return

        try:
            self.client = OpenAI(base_url=api_base, api_key=api_key)
            self.available = True
        except Exception as e:
            print(f"LLM 客户端初始化失败: {e}", file=sys.stderr)

    def _call(self, prompt, retries=None):
        """调用 LLM 并返回 JSON 解析结果"""
        if not self.available:
            return None

        retries = retries or self.max_retries
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=500,
                )
                text = resp.choices[0].message.content.strip()
                # 提取 JSON
                match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
                if match:
                    return json.loads(match.group())
                return json.loads(text)
            except json.JSONDecodeError:
                if attempt < retries - 1:
                    time.sleep(self.retry_delay)
                continue
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    print(f"LLM 调用失败: {e}", file=sys.stderr)
                continue
        return None

    def evaluate_single(self, item):
        """评估单个样本，返回 4 维度评分 dict"""
        info = _extract_info(item)
        question = _extract_question(item)
        answer = _extract_full_output(item)

        prompt = SINGLE_EVAL_PROMPT.format(
            info=info[:1500],
            question=question[:500],
            answer=answer[:2000],
        )
        return self._call(prompt)

    def evaluate_batch(self, data, progress=True):
        """批量评估，返回评分列表"""
        results = []
        total = len(data)
        for idx, item in enumerate(data):
            if progress and (idx + 1) % 10 == 0:
                print(f"  LLM 评估进度: {idx + 1}/{total}", file=sys.stderr)
            result = self.evaluate_single(item)
            results.append(result)
            time.sleep(0.5)  # Rate limiting
        return results

    def pairwise(self, item_a, item_b, name_a="A", name_b="B"):
        """Pairwise 对决，返回 winner 和分数"""
        # 随机化顺序以避免位置偏差
        swap = random.random() < 0.5

        if swap:
            item_a, item_b = item_b, item_a
            name_a, name_b = name_b, name_a

        prompt = PAIRWISE_PROMPT.format(
            info_a=_extract_info(item_a)[:1000],
            question_a=_extract_question(item_a)[:300],
            answer_a=_extract_full_output(item_a)[:1500],
            info_b=_extract_info(item_b)[:1000],
            question_b=_extract_question(item_b)[:300],
            answer_b=_extract_full_output(item_b)[:1500],
        )

        result = self._call(prompt)
        if result and swap:
            # 还原顺序
            w = result.get("winner", "tie")
            if w == "A":
                result["winner"] = "B"
            elif w == "B":
                result["winner"] = "A"
            result["score_a"], result["score_b"] = result.get("score_b", 0), result.get("score_a", 0)

        return result

    def pairwise_batch(self, data_a, data_b, name_a="SoG", name_b="Doc", progress=True):
        """批量 Pairwise 对决"""
        results = []
        n = min(len(data_a), len(data_b))
        for idx in range(n):
            if progress and (idx + 1) % 10 == 0:
                print(f"  Pairwise 进度: {idx + 1}/{n}", file=sys.stderr)
            result = self.pairwise(data_a[idx], data_b[idx], name_a, name_b)
            results.append(result)
            time.sleep(0.5)
        return results


# ════════════════════════════════════════════════════════════
# 辅助函数 (独立于 evaluate.py，避免循环导入)
# ════════════════════════════════════════════════════════════

def _extract_info(item):
    inp = item["input"]
    for sep in ["### 信息:", "### 信息："]:
        if sep in inp:
            info = inp.split(sep, 1)[1]
            for qsep in ["### 问题:", "### 问题："]:
                if qsep in info:
                    info = info.split(qsep, 1)[0]
            return info.strip()
    return inp.strip()


def _extract_question(item):
    inp = item["input"]
    for sep in ["### 问题:", "### 问题："]:
        if sep in inp:
            return inp.split(sep, 1)[1].strip()
    return ""


def _extract_full_output(item):
    return item["output"]


# ════════════════════════════════════════════════════════════
# 独立运行入口
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-as-Judge 独立评估")
    parser.add_argument("--data", required=True, help="数据文件 (JSON)")
    parser.add_argument("--name", default="Method", help="方法名称")
    parser.add_argument("--n", type=int, default=20, help="评估样本数")
    parser.add_argument("--config", help="配置文件 (YAML)")

    # Pairwise 模式
    parser.add_argument("--data-b", help="对比数据 (JSON)")
    parser.add_argument("--name-b", default="Baseline", help="对比方法名称")
    parser.add_argument("--pairwise-n", type=int, default=20, help="Pairwise 对数")

    args = parser.parse_args()

    config = None
    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"配置加载失败: {e}")

    judge = LLMJudge(config)
    if not judge.available:
        print("错误: LLM 未配置。请设置环境变量 LLM_API_KEY 或提供 config.yaml")
        sys.exit(1)

    print(f"模型: {judge.model}")

    with open(args.data, encoding="utf-8") as f:
        data_a = json.load(f)

    import numpy as np

    if args.data_b:
        # Pairwise 模式
        with open(args.data_b, encoding="utf-8") as f:
            data_b = json.load(f)

        rng = random.Random(42)
        sample_a = rng.sample(data_a, min(args.pairwise_n, len(data_a)))
        sample_b = rng.sample(data_b, min(args.pairwise_n, len(data_b)))

        print(f"\nPairwise 对决: {args.name} vs {args.name_b} ({len(sample_a)} 对)")
        results = judge.pairwise_batch(sample_a, sample_b, args.name, args.name_b)

        a_wins = sum(1 for r in results if r and r.get("winner") == "A")
        b_wins = sum(1 for r in results if r and r.get("winner") == "B")
        ties = sum(1 for r in results if r and r.get("winner") == "tie")

        print(f"\n{args.name} 胜: {a_wins}")
        print(f"{args.name_b} 胜: {b_wins}")
        print(f"平局: {ties}")

    else:
        # 单样本评估
        rng = random.Random(42)
        sample = rng.sample(data_a, min(args.n, len(data_a)))

        print(f"\n评估 {args.name}: {len(sample)} 条样本")
        results = judge.evaluate_batch(sample)

        valid = [r for r in results if r]
        if valid:
            dims = ["回答质量", "推理质量", "知识利用", "训练价值"]
            for d in dims:
                avg = np.mean([r.get(d, 0) for r in valid])
                print(f"  {d}: {avg:.2f}")
            overall = np.mean([np.mean(list(r.values())) for r in valid])
            print(f"  综合均分: {overall:.2f}")
        else:
            print("  无有效结果")
