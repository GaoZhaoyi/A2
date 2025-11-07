from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from transformers import DataCollatorForSeq2Seq

from constants import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset using streaming to avoid compatibility issues.
    """
    # 使用streaming方式加载数据集，避免pickle问题
    try:
        # 首先尝试正常加载
        dataset = load_dataset("wmt19", "zh-en")
        train_dataset = dataset["train"].select(range(100000))
        validation_dataset = dataset["train"].select(range(100000, 102000))
        test_dataset = dataset["validation"]
    except Exception as e:
        # 如果正常加载失败，使用流式加载
        print("Using streaming mode due to compatibility issues...")
        dataset = load_dataset("wmt19", "zh-en", streaming=True)

        # 收集训练数据
        train_samples = []
        for i, sample in enumerate(dataset["train"]):
            if i >= 100000:
                break
            train_samples.append(sample)

        # 收集验证数据
        validation_samples = []
        skip_count = 0
        for i, sample in enumerate(dataset["train"]):
            if i >= 100000 and i < 102000:
                validation_samples.append(sample)
            elif i >= 102000:
                break

        # 转换为Dataset对象
        train_dataset = Dataset.from_list(train_samples)
        validation_dataset = Dataset.from_list(validation_samples)

        # 测试集
        test_samples = []
        for sample in dataset["validation"]:
            test_samples.append(sample)
        test_dataset = Dataset.from_list(test_samples)

    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def preprocess_function(examples, prefix, tokenizer, max_input_length, max_target_length):
    """
    Preprocess the data.
    """
    inputs = [prefix + ex["zh"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_data(raw_datasets: DatasetDict, tokenizer) -> DatasetDict:
    """
    Preprocess the data.
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
