from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from transformers import DataCollatorForSeq2Seq

from constants import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset.

    Returns:
        The dataset.

    NOTE: You can replace this with your own dataset. Make sure to include
    the `validation` split and ensure that it is the same as the test split from the WMT19 dataset,
    Which means that:
        raw_datasets["validation"] = load_dataset('wmt19', 'zh-en', split="validation")
    """
    dataset = load_dataset("wmt19", "zh-en")        # 从Hugging Face Hub加载WMT19中文-英文翻译数据集
    train_dataset = dataset["train"].select(range(100000)) # 从训练集中选择前130万个样本
    validation_dataset = dataset["train"].select(range(100000, 102000)) # 选择1300000到1302000的样本作为验证集

    # NOTE: You should not change the test dataset
    test_dataset = dataset["validation"]    # 使用原始的验证集作为测试集
    # 将数据集组织成字典格式，包含训练、验证、测试三个部分
    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks.

    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        DataCollatorForSeq2Seq instance.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def preprocess_function(examples, prefix, tokenizer, max_input_length, max_target_length):
    """
    Preprocess the data.

    Args:
        examples: Examples.
        prefix: Prefix.
        tokenizer: Tokenizer object.
        max_input_length: Maximum input length.
        max_target_length: Maximum target length.

    Returns:
        Model inputs.
    """
    inputs = [prefix + ex["zh"] for ex in examples["translation"]]      # 提取中文句子
    targets = [ex["en"] for ex in examples["translation"]]              # 提取英文句子

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)      # 对中文句子进行分词处理
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)  # 对英文句子进行分词处理

    model_inputs["labels"] = labels["input_ids"]    # 将英文句子作为标签，用于训练时的损失计算
    return model_inputs


def preprocess_data(raw_datasets: DatasetDict, tokenizer) -> DatasetDict:
    """
    Preprocess the data.

    Args:
        raw_datasets: Raw datasets.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized datasets.
    """
    tokenized_datasets: DatasetDict = raw_datasets.map(
        function=lambda examples: preprocess_function(
            examples=examples,
            prefix="",
            tokenizer=tokenizer,
            max_input_length=MAX_INPUT_LENGTH,
            max_target_length=MAX_TARGET_LENGTH,
        ),
        batched=True,
    )
    return tokenized_datasets
