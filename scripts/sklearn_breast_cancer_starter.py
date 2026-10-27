"""
Sklearn starter: import breast-cancer dataset tooling and ML models.

Run from the repo root:
    python scripts/sklearn_breast_cancer_starter.py
"""


def main() -> None:
    # 1. 导入数据集和机器学习模型
    from sklearn.datasets import load_breast_cancer  # noqa: F401
    from sklearn.model_selection import train_test_split  # noqa: F401
    from sklearn.ensemble import RandomForestClassifier  # noqa: F401
    from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: F401

    # 2. 导入评估和数据整理工具
    from sklearn.metrics import accuracy_score  # noqa: F401
    import pandas as pd
    import matplotlib.pyplot as plt  # noqa: F401

    # 让表格和图表更容易阅读
    pd.set_option("display.max_columns", None)

    print("All libraries imported successfully!")


if __name__ == "__main__":
    main()
