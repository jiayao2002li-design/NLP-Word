"""
Sklearn starter: breast-cancer Wisconsin load + inspect.

Run from the repo root:
    python scripts/sklearn_breast_cancer_starter.py
"""


def main() -> None:
    # 1. 导入数据集和机器学习模型
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: F401

    # 2. 导入评估和数据整理工具
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import matplotlib.pyplot as plt  # noqa: F401

    try:
        from IPython.display import display
    except ImportError:
        display = print

    # 让表格和图表更容易阅读
    pd.set_option("display.max_columns", None)

    print("All libraries imported successfully!")

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


if __name__ == "__main__":
    main()
