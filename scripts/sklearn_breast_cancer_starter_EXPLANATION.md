# Step-by-Step Explanation: Sklearn Surrogate Tree Starter

This document explains what `scripts/sklearn_breast_cancer_starter.py` does, in **English** and **Chinese**. The Python code itself is left unchanged.

**Project note:** The official **project benchmark** for this work is tabular Breast Cancer Wisconsin (not NLP). Prefer:

```bash
python scripts/breast_cancer_surrogate_benchmark.py
```

Updated research conclusion: [`BREAST_CANCER_BENCHMARK_CONCLUSION.md`](BREAST_CANCER_BENCHMARK_CONCLUSION.md).

Starter script:

```bash
python scripts/sklearn_breast_cancer_starter.py
```

---

## Overview / 概览

**English:** The script trains a Random Forest on the Breast Cancer Wisconsin dataset, then builds simpler **surrogate decision trees** that try to copy the forest’s predictions. It measures how well those trees match the forest (**fidelity**), how accurate they are vs. true labels, how complex their rules are, plots fidelity vs. depth, prints one example rule set, and saves a CSV of results.

**中文：** 脚本先在 Breast Cancer Wisconsin 数据集上训练随机森林，再用更简单的**替代决策树**去模仿森林的预测。接着比较替代树与森林的一致程度（**保真度 / fidelity**）、相对真实标签的准确率、规则复杂度，绘制保真度随深度变化的图，打印一棵示例规则树，并把结果表保存为 CSV。

---

## Step 1: Imports and display helpers

**English:** The script imports sklearn dataset/model tools (`load_breast_cancer`, `train_test_split`, `RandomForestClassifier`, `DecisionTreeClassifier`, `export_text`), metrics (`accuracy_score`), plus `pandas` and `matplotlib`. It tries to use IPython’s `display` (nice in notebooks); if that import fails, it falls back to `print`. `pd.set_option("display.max_columns", None)` so wide tables are not truncated.

**中文：** 导入 sklearn 的数据与模型工具（`load_breast_cancer`、`train_test_split`、`RandomForestClassifier`、`DecisionTreeClassifier`、`export_text`）、评估指标（`accuracy_score`），以及 `pandas` 和 `matplotlib`。优先使用 IPython 的 `display`（笔记本里更美观）；导入失败则退回 `print`。`pd.set_option("display.max_columns", None)` 让宽表列不被截断。

---

## Step 2: Load the Breast Cancer Wisconsin dataset

**English:** `load_breast_cancer()` returns feature matrix and labels. Features go into DataFrame `X` (named columns); labels are `y` (0 = malignant, 1 = benign in sklearn’s encoding via `target_names`). The script prints shape, feature count, class names, the first 5 rows, and how many samples belong to each class.

**中文：** `load_breast_cancer()` 返回特征与标签。特征放进带列名的 DataFrame `X`；标签为 `y`（sklearn 中通过 `target_names`：0 为 malignant，1 为 benign）。脚本打印数据形状、特征数、类别名、前 5 行，以及各类样本数量。

---

## Step 3: Train / test split

**English:** `train_test_split` holds out 30% as the test set (`test_size=0.30`), with `random_state=42` for reproducibility and `stratify=y` so train and test keep similar class proportions. It prints how many samples landed in train vs. test (about 398 / 171).

**中文：** `train_test_split` 留出 30% 作为测试集（`test_size=0.30`），`random_state=42` 保证可复现，`stratify=y` 让训练/测试的类别比例接近。并打印训练集与测试集样本数（约 398 / 171）。

---

## Step 4: Train the Random Forest (teacher model)

**English:** A `RandomForestClassifier` with 100 trees (`n_estimators=100`) is fit on **true labels** (`y_train`). It predicts on train and test (`rf_train_pred`, `rf_test_pred`). Test **accuracy** vs. true `y_test` is printed — this is the “black-box” model we will try to explain with simpler trees.

**中文：** 用 100 棵树的 `RandomForestClassifier`（`n_estimators=100`）在**真实标签** `y_train` 上训练。再对训练集和测试集做预测（`rf_train_pred`、`rf_test_pred`）。打印相对真实 `y_test` 的测试**准确率**——这是后面要用更简单的树去解释的“黑盒”教师模型。

---

## Step 5: Surrogate decision trees (student models)

**English:** For each max depth in `[1, 2, 3, 4, 5, 6]`, the script fits a `DecisionTreeClassifier` on `X_train` but with targets **`rf_train_pred` (the forest’s labels), not `y_train`**. That is the key idea of a **surrogate / student** model: imitate the teacher. Each tree is stored in `surrogate_models`. On the test set it records:

| Metric | Meaning |
|---|---|
| **Fidelity** | Fraction of test predictions that match the Random Forest |
| **Accuracy** | Fraction that match the true labels `y_test` |
| **Actual tree depth** | Depth the tree actually reached |
| **Number of rules** | Number of leaves (one path/rule per leaf) |

**中文：** 对最大深度 `[1, 2, 3, 4, 5, 6]` 各训练一棵 `DecisionTreeClassifier`：输入仍是 `X_train`，但目标是 **`rf_train_pred`（森林的预测），不是真实 `y_train`**。这就是**替代 / 学生模型**的核心：模仿教师。每棵树存入 `surrogate_models`。在测试集上记录：

| 指标 | 含义 |
|---|---|
| **Fidelity（保真度）** | 与随机森林预测一致的比例 |
| **Accuracy（准确率）** | 与真实标签 `y_test` 一致的比例 |
| **Actual tree depth** | 树实际达到的深度 |
| **Number of rules** | 叶节点数（每条根到叶路径可看作一条规则） |

---

## Step 6: Average rule length

**English:** `get_rule_lengths` walks the fitted tree from the root. At each split it goes left and right, increasing depth by 1; at a leaf it records that path length (number of conditions). Averaging those lengths for each surrogate gives **Average rule length** — a complexity measure (shallower / shorter paths ≈ simpler explanations). The enriched table is displayed again.

**中文：** `get_rule_lengths` 从根节点遍历树：遇到分裂就分别走左右子树并把深度 +1；到达叶节点则记录该路径长度（条件个数）。对各替代树求平均，得到 **Average rule length（平均规则长度）**——复杂度指标（路径更短 ≈ 解释更简单）。带新列的结果表会再展示一次。

---

## Step 7: Fidelity vs. depth plot

**English:** Matplotlib plots actual tree depth on the x-axis and fidelity on the y-axis (markers on a line). This shows the usual trade-off: deeper trees can follow the forest more closely, but rules get more complex. `plt.show()` opens the figure when a GUI/notebook backend is available.

**中文：** 用 matplotlib 画图：横轴为实际树深度，纵轴为保真度（折线加点）。用来观察常见权衡：树越深通常越能贴近森林，但规则也更复杂。有图形界面或笔记本后端时，`plt.show()` 会弹出图像。

---

## Step 8: Print example rules (depth = 3)

**English:** The surrogate with `max_depth=3` is turned into readable if/then text via `export_text`, using the dataset’s feature names. Printed rules look like “if worst radius ≤ 16.80 and … then class 0/1”, so you can inspect how the student approximates the forest.

**中文：** 取出 `max_depth=3` 的替代树，用 `export_text` 和特征名转成可读的 if/then 文本。输出形如“若 worst radius ≤ 16.80 且 … 则类别 0/1”，便于人工查看学生模型如何近似森林。

---

## Step 9: Save CSV (and optional Colab download)

**English:** `results_df` is written to `surrogate_tree_results.csv` in the current working directory. If running inside Google Colab, `files.download` triggers a browser download; outside Colab the import fails and the script prints that download was skipped (local runs still have the CSV on disk).

**中文：** 将 `results_df` 写入当前工作目录下的 `surrogate_tree_results.csv`。若在 Google Colab 中运行，`files.download` 会触发浏览器下载；在本地则导入失败并提示跳过下载（CSV 仍已保存在磁盘上）。

---

## Big picture / 一句话总结

**English:** Random Forest = accurate but hard to read → shallow decision trees trained on the forest’s predictions = interpretable approximations → compare fidelity, accuracy, and rule complexity across depths.

**中文：** 随机森林 = 准但难读 → 用森林预测训练的浅层决策树 = 可解释的近似 → 在不同深度下比较保真度、准确率与规则复杂度。
