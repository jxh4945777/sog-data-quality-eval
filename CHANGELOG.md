# Changelog

## v2.0 (2026-03-24)

### 新增
- **LLM-as-Judge 模块** (`llm_judge.py`): 支持 4 维度单样本质量打分
- **Pairwise 对决**: 样本对直接对比评判，含位置随机化消除偏差
- **加权综合评分**: 基于指标权重的百分制综合得分
- **灵活 LLM 配置**: 支持 OpenAI API / 自定义中转站 / 本地部署 (vLLM, Ollama, llama.cpp)
- 新增 4 个指标: ROUGE-L 忠实度 (M02)、源文档 Trigram 覆盖 (M10)、多段落引用率 (M14)、步骤均衡度 CV (M17)

### 变更
- 评估框架从 7 维度 13 指标升级为 **8 维度 17 指标**
- 新增忠实性 (B) 维度，提升评估平衡性和客观性
- CLI 新增 `--llm-judge`、`--pairwise`、`--llm-config` 参数

### 改进
- 对比报告增加维度级别胜负统计
- 评估报告增加学术出处和方法说明
- 添加 `.gitignore`、`LICENSE`、完善 `README.md`

## v1.0 (2026-03-23)

### 初始版本
- 7 维度 13 个客观评估指标
- 支持单方法评估和双方法对比
- 基于 RAGAS / Dingo / Data-Juicer / distilabel 方法论
- 等量采样，随机种子固定，结果可复现
