# Step-by-Step Explanation: BERT Word Filter Model

This document explains what each step of the code does, in **English** and **Chinese**.

---

## Step 1: Load the Dataset

**English:** The script loads the Jigsaw toxicity dataset from Hugging Face using `datasets.load_dataset("google/jigsaw_toxicity_pred")`. This dataset contains Wikipedia comments with human-labeled toxicity. The training split is then divided into a training set and a validation set (e.g. 90% train, 10% validation) so we can monitor performance during training without touching the official test set. This happens in `train.py` in the function `load_and_prepare_data()`.

**中文：** 脚本通过 `datasets.load_dataset("google/jigsaw_toxicity_pred")` 从 Hugging Face 加载 Jigsaw 毒性数据集。该数据集包含带有毒性人工标注的维基百科评论。训练集被进一步划分为训练子集和验证子集（例如 90% 训练、10% 验证），以便在训练过程中评估模型表现，而不使用官方测试集。对应代码在 `train.py` 的 `load_and_prepare_data()` 中。

---

## Step 2: Prepare the Labels

**English:** The dataset has six binary columns: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. Each comment can have multiple labels (multi-label classification). The code collects these six columns into a single array of shape (number of samples, 6) with float values 0.0 or 1.0. This array is used as the target for the model. The helper `get_labels_array()` in `train.py` builds this label matrix from the dataset.

**中文：** 数据集有六个二值列：`toxic`、`severe_toxic`、`obscene`、`threat`、`insult`、`identity_hate`。一条评论可以同时属于多个类别（多标签分类）。代码把这六列合并成一个形状为（样本数, 6）的数组，取值为 0.0 或 1.0，作为模型的目标。在 `train.py` 中由 `get_labels_array()` 从数据集生成该标签矩阵。

---

## Step 3: Tokenize with BERT

**English:** BERT expects integer token IDs, not raw text. The tokenizer converts each comment into a fixed-length sequence of token IDs (`input_ids`) and an `attention_mask` (1 for real tokens, 0 for padding). We use a maximum length (e.g. 128 tokens) and truncate longer comments or pad shorter ones. This is done in `train.py` via `tokenize_examples()` and the `.map()` call on the dataset; the same tokenizer and max length are defined in `config.py`.

**中文：** BERT 需要的是整数 token ID，而不是原始文本。分词器把每条评论转换成固定长度的 token ID 序列（`input_ids`）和注意力掩码（`attention_mask`，真实 token 为 1，填充为 0）。我们设定最大长度（如 128），过长则截断，过短则填充。在 `train.py` 中通过 `tokenize_examples()` 以及数据集的 `.map()` 完成；分词器和最大长度在 `config.py` 中配置。

---

## Step 4: Build the Model

**English:** We use `BertForSequenceClassification` with `num_labels=6`. The model has BERT as the backbone and a linear layer on top that outputs six logits (one per toxicity type). For multi-label classification we do not use softmax; instead we apply sigmoid to each logit and use binary cross-entropy loss (BCEWithLogitsLoss). The model and tokenizer are loaded from the pre-trained `bert-base-uncased` checkpoint in `train.py` inside the `train()` function.

**中文：** 我们使用 `BertForSequenceClassification` 并设置 `num_labels=6`。模型以 BERT 为骨干，顶层线性层输出六个 logit（对应六种毒性类型）。多标签分类不用 softmax，而是对每个 logit 做 sigmoid，并用二元交叉熵损失（BCEWithLogitsLoss）。在 `train.py` 的 `train()` 中从预训练的 `bert-base-uncased` 加载模型和分词器。

---

## Step 5: Train

**English:** The training loop iterates over batches of tokenized inputs and labels. For each batch we run a forward pass, compute BCE loss between the model’s logits and the true multi-label vector, then backpropagate and update weights with AdamW. We use a linear learning-rate schedule with warmup and gradient clipping to stabilize training. Training runs for several epochs; after each epoch we can print the average training loss. All of this is implemented in the `train()` function in `train.py`.

**中文：** 训练循环按批处理分词后的输入和标签。对每个批次做前向传播，计算模型 logit 与真实多标签向量之间的 BCE 损失，然后反向传播并用 AdamW 更新参数。我们使用带 warmup 的线性学习率调度和梯度裁剪以稳定训练。训练进行若干 epoch；每个 epoch 后可打印平均训练损失。这些都在 `train.py` 的 `train()` 中实现。

---

## Step 6: Evaluate and Save

**English:** After each epoch (or at the end of training), we evaluate on the validation set: we compute validation loss and metrics such as micro F1, macro F1, and ROC-AUC for the six labels. This tells us how well the model generalizes. When training is done, we save the fine-tuned model and tokenizer to a folder (e.g. `./saved_model`) using `save_pretrained()`, so they can be loaded later for inference. The evaluation logic is in `evaluate()` in `train.py`; saving is at the end of `train()`.

**中文：** 在每个 epoch 结束后（或训练结束时），在验证集上评估：计算验证损失以及六类标签的 micro F1、macro F1 和 ROC-AUC，用于衡量模型泛化能力。训练结束后，用 `save_pretrained()` 将微调后的模型和分词器保存到指定目录（如 `./saved_model`），便于之后做推理。评估逻辑在 `train.py` 的 `evaluate()` 中；保存逻辑在 `train()` 末尾。

---

## Step 7: Run Inference (Word Filter)

**English:** The inference script `predict.py` loads the saved model and tokenizer from disk. Given new text (one string or a list of strings), it tokenizes the text, runs the model forward, and applies sigmoid to the logits to get probabilities for each of the six toxicity types. The function `is_toxic()` returns a single boolean: True if any of the six probabilities is above a threshold (default 0.5). The function `get_toxic_labels()` returns the names of the labels that exceed the threshold. This is how we use the model as a “word filter” to flag toxic content.

**中文：** 推理脚本 `predict.py` 从磁盘加载保存的模型和分词器。给定新文本（一条或若干条），对文本分词、前向计算，并对 logit 做 sigmoid 得到六类毒性的概率。函数 `is_toxic()` 返回一个布尔值：若六个概率中有任意一个超过阈值（默认 0.5）则为 True。函数 `get_toxic_labels()` 返回超过阈值的标签名称。这样即可将模型用作“敏感词/毒性过滤器”来标记有害内容。
