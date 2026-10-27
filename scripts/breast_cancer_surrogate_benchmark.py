"""
Project benchmark (tabular medical — NOT NLP): Random Forest + surrogate trees
on Breast Cancer Wisconsin.

Teacher = RandomForestClassifier on the 30 clinical features.
Students = DecisionTreeClassifier models that imitate the forest (rule-based
explanations). Evaluation follows the common literature protocol: fidelity +
complexity (rule count / depth / average rule length), plus accuracy vs labels.

Run from the repo root:
    python scripts/breast_cancer_surrogate_benchmark.py
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    # 1. 导入数据集和机器学习模型
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text

    # 2. 导入评估和数据整理工具
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import matplotlib.pyplot as plt

    try:
        from IPython.display import display
    except ImportError:

        def display(obj):
            data = getattr(obj, "data", obj)
            print(data)

    # 让表格和图表更容易阅读
    pd.set_option("display.max_columns", None)

    print("All libraries imported successfully!")
    print("Benchmark dataset: Breast Cancer Wisconsin (sklearn) — tabular, not NLP")

    # 加载 Breast Cancer Wisconsin 数据集
    data = load_breast_cancer()

    # X 是输入特征，y 是真实分类标签
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    print("Dataset shape:", X.shape)
    print("Number of features:", X.shape[1])
    print("Class names:", data.target_names)
    print("\nFirst 5 rows:")
    display(X.head())

    print("\nClass counts:")
    print(
        pd.Series(y)
        .value_counts()
        .rename(
            index={
                0: data.target_names[0],
                1: data.target_names[1],
            }
        )
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    # 使用真实标签训练原始模型
    rf.fit(X_train, y_train)

    # Random Forest 对训练集和测试集的预测
    rf_train_pred = rf.predict(X_train)
    rf_test_pred = rf.predict(X_test)

    # 原始模型对真实标签的测试准确率
    rf_accuracy = accuracy_score(y_test, rf_test_pred)

    print(f"Random Forest test accuracy: {rf_accuracy:.4f}")

    # 想比较的最大树深度
    depth_values = [1, 2, 3, 4, 5, 6]

    results = []
    surrogate_models = {}

    for depth in depth_values:
        # 创建指定最大深度的替代决策树
        surrogate = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42,
        )

        # 重点：学习 Random Forest 的训练集预测，而不是真实标签
        surrogate.fit(X_train, rf_train_pred)

        # 保存模型，方便之后检查规则
        surrogate_models[depth] = surrogate

        # 在测试集上进行预测
        surrogate_test_pred = surrogate.predict(X_test)

        # Fidelity：替代树与 Random Forest 的预测一致比例
        fidelity = (surrogate_test_pred == rf_test_pred).mean()

        # Accuracy：替代树与数据集真实标签的一致比例
        accuracy = accuracy_score(y_test, surrogate_test_pred)

        # 复杂度指标
        actual_depth = surrogate.get_depth()
        number_of_rules = surrogate.get_n_leaves()

        results.append(
            {
                "Max depth setting": depth,
                "Actual tree depth": actual_depth,
                "Number of rules": number_of_rules,
                "Fidelity": fidelity,
                "Accuracy": accuracy,
            }
        )

    results_df = pd.DataFrame(results)

    format_cols = {
        "Fidelity": "{:.4f}",
        "Accuracy": "{:.4f}",
        "Average rule length": "{:.2f}",
    }

    def show_results_table(df):
        try:
            display(
                df.style.format(
                    {k: v for k, v in format_cols.items() if k in df.columns}
                )
            )
        except (AttributeError, ImportError):
            print(
                df.to_string(
                    formatters={
                        k: v.format for k, v in format_cols.items() if k in df.columns
                    }
                )
            )

    show_results_table(results_df)

    def get_rule_lengths(tree_model):
        """
        计算决策树中每条根到叶路径的长度。
        每经过一个分裂节点，就增加一个条件。
        """
        tree = tree_model.tree_
        rule_lengths = []

        def visit(node, depth):
            left = tree.children_left[node]
            right = tree.children_right[node]

            # 左右子节点都不存在，说明当前节点是叶节点
            is_leaf = left == right

            if is_leaf:
                rule_lengths.append(depth)
            else:
                visit(left, depth + 1)
                visit(right, depth + 1)

        visit(0, 0)
        return rule_lengths

    # 为每个模型计算平均规则长度
    avg_lengths = []

    for depth in depth_values:
        model = surrogate_models[depth]
        lengths = get_rule_lengths(model)
        avg_lengths.append(sum(lengths) / len(lengths))

    results_df["Average rule length"] = avg_lengths

    show_results_table(results_df)

    plt.figure(figsize=(7, 5))

    plt.plot(
        results_df["Actual tree depth"],
        results_df["Fidelity"],
        marker="o",
        label="Fidelity vs RF",
    )
    plt.plot(
        results_df["Actual tree depth"],
        results_df["Accuracy"],
        marker="s",
        label="Accuracy vs true label",
    )

    plt.xlabel("Actual tree depth")
    plt.ylabel("Score")
    plt.title("Breast Cancer: Fidelity & Accuracy vs Surrogate Depth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(REPO_ROOT, "breast_cancer_surrogate_fidelity_depth.png")
    plt.savefig(plot_path, dpi=120)
    print(f"Plot saved to {plot_path}")
    plt.show()

    example_depth = 3
    example_model = surrogate_models[example_depth]

    rules_text = export_text(
        example_model,
        feature_names=list(X.columns),
        decimals=2,
    )

    print(f"\nExample rules (max_depth={example_depth}):\n")
    print(rules_text)

    # 保存实验结果表格
    out_csv = os.path.join(REPO_ROOT, "breast_cancer_surrogate_tree_results.csv")
    results_df.to_csv(out_csv, index=False)

    print(f"Results saved to {out_csv}")

    # Colab download (optional)
    try:
        from google.colab import files

        files.download(out_csv)
    except ImportError:
        print("Skipping Colab download (not running in Google Colab).")

    majority_acc = max(float((y_test == 0).mean()), float((y_test == 1).mean()))
    best_fid = results_df.loc[results_df["Fidelity"].idxmax()]
    best_acc = results_df.loc[results_df["Accuracy"].idxmax()]
    sparsest = results_df.loc[results_df["Number of rules"].idxmin()]

    print("\n=== Breast Cancer benchmark summary ===")
    print(
        f"RF teacher — accuracy={rf_accuracy:.4f}, "
        f"majority_baseline_acc={majority_acc:.4f}, "
        f"n_train={len(X_train)}, n_test={len(X_test)}, "
        f"features={X.shape[1]}"
    )
    print(
        f"Best fidelity: depth={int(best_fid['Max depth setting'])}, "
        f"fid={best_fid['Fidelity']:.4f}, acc={best_fid['Accuracy']:.4f}, "
        f"rules={int(best_fid['Number of rules'])}"
    )
    print(
        f"Best accuracy: depth={int(best_acc['Max depth setting'])}, "
        f"fid={best_acc['Fidelity']:.4f}, acc={best_acc['Accuracy']:.4f}, "
        f"rules={int(best_acc['Number of rules'])}"
    )
    print(
        f"Sparsest: depth={int(sparsest['Max depth setting'])}, "
        f"fid={sparsest['Fidelity']:.4f}, acc={sparsest['Accuracy']:.4f}, "
        f"rules={int(sparsest['Number of rules'])}"
    )
    winners = {
        int(best_fid["Max depth setting"]),
        int(best_acc["Max depth setting"]),
    }
    print(f"Fidelity / accuracy agree on one depth: {len(winners) == 1}")
    print(
        "Conclusion cue: sparsity≠best on all metrics "
        f"(sparsest depth={int(sparsest['Max depth setting'])}, "
        f"best-acc depth={int(best_acc['Max depth setting'])}, "
        f"best-fid depth={int(best_fid['Max depth setting'])})"
    )


if __name__ == "__main__":
    main()
