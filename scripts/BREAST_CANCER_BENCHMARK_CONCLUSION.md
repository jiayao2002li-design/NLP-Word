# Updated Conclusion — Breast Cancer Wisconsin Surrogate Benchmark

**Scope:** Tabular medical classification (Breast Cancer Wisconsin). **Not an NLP project.**  
**Run:** `python scripts/breast_cancer_surrogate_benchmark.py`  
**Artifacts:** `breast_cancer_surrogate_tree_results.csv`, `breast_cancer_surrogate_fidelity_depth.png`

---

## Retrained results (random_state=42, 70/30 stratified)

| Item | Value |
|---|---|
| Samples / features | 569 / 30 |
| Train / test | 398 / 171 |
| RF teacher test accuracy | **0.9357** |
| Majority-class baseline | 0.6257 |

| Max depth | Rules | Fidelity (vs RF) | Accuracy (vs label) | Avg rule length |
|---:|---:|---:|---:|---:|
| 1 | 2 | **0.9649** | 0.9123 | 1.00 |
| 2 | 4 | 0.9591 | 0.9181 | 2.00 |
| 3 | 7 | 0.9532 | 0.9240 | 2.86 |
| 4 | 12 | 0.9532 | 0.9240 | 3.75 |
| 5 | 14 | 0.9474 | **0.9298** | 4.07 |
| 6 | 16 | 0.9474 | 0.9181 | 4.44 |

- **Best fidelity:** depth 1 (2 rules)  
- **Best accuracy:** depth 5 (14 rules)  
- **Sparsest:** depth 1  
- **Fidelity and accuracy do not pick the same depth**

---

## English — updated research conclusion

On this **Breast Cancer Wisconsin** benchmark (tabular clinical features, Random Forest teacher, decision-tree surrogates trained on the forest’s predictions):

1. **Multi-dimensional assessment is necessary.** Optimizing fidelity alone selects depth 1; optimizing accuracy vs. true diagnosis selects depth 5. A single score would misrepresent “best” explanation quality.
2. **The literature’s fidelity + rule-count protocol is reproducible here**, but it is not enough by itself: the sparsest model wins fidelity while losing ground-truth accuracy.
3. **Computational sparsity does not guarantee reliable reasoning.** Depth 1 (2 rules) is easiest to read and closest to the RF on this test split, yet it is the weakest vs. true labels among depths 1–5. Adding rules (up to depth 5) improves label accuracy without improving fidelity.
4. **What this benchmark does *not* claim:** it does not replace a multi-dataset standardized suite, and it does not measure structural trustworthiness (rule overlap / logical redundancy). Those remain open gaps relative to the slide.

**Bottom line:** For rule-based explanations of a Random Forest on Breast Cancer Wisconsin, **fewer rules ≠ better explanation across all criteria**. Evaluation must report at least fidelity, label accuracy, and complexity together.

---

## 中文 — 更新后的研究结论

在本 **Breast Cancer Wisconsin（威斯康星乳腺癌）** 基准（表格型临床特征；随机森林为教师；决策树替代模型学习森林预测）上重新训练后：

1. **必须多维度评估。** 只看保真度会选 depth 1；只看相对真实诊断的准确率会选 depth 5。单一指标无法代表“最好的解释”。
2. **文献常用的保真度 + 规则数量协议可以在此复现**，但不够：最稀疏的模型保真度最高，相对真实标签的准确率却更差。
3. **计算上的稀疏性不等于可靠推理。** Depth 1（2 条规则）最好读、也最贴近森林，但在 depth 1–5 中相对真标签最弱；增至 depth 5 时准确率上升，保真度并未上升。
4. **本基准不声称：** 不能替代多数据集标准评测协议；也未测量结构可信度（规则重叠 / 逻辑冗余）。相对幻灯片中的缺口，这两点仍待补齐。

**一句话：** 在乳腺癌表格数据上解释随机森林时，**更少规则 ≠ 在所有标准上都更好的解释**；至少应同时报告保真度、标签准确率与复杂度。
