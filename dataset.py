from datasets import Dataset, DatasetDict, load_dataset
from transformers import DataCollatorForSeq2Seq

from constants import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH


def build_dataset() -> DatasetDict:
    """
    Build the dataset using TED Talks which has high quality translations.
    """
    print("Loading TED Talks dataset...")

    try:
        # 使用TED Talks数据集，质量更高
        dataset = load_dataset("ted_talks_iwslt", "zh-en")

        # 使用完整的训练集
        train_dataset = dataset["train"]
        # 限制训练数据量以适应时间约束
        if len(train_dataset) > 100000:
            train_dataset = train_dataset.select(range(100000))

        # 验证集
        validation_dataset = dataset["validation"]
        if len(validation_dataset) > 2000:
            validation_dataset = validation_dataset.select(range(2000))

        # 测试集
        test_dataset = dataset["test"]

        print(f"Train size: {len(train_dataset)}")
        print(f"Validation size: {len(validation_dataset)}")
        print(f"Test size: {len(test_dataset)}")

    except Exception as e:
        print(f"Error loading TED Talks dataset: {e}")
        print("Falling back to a smaller high-quality dataset...")

        # 回退到opus100的一个子集
        dataset = load_dataset("opus100", "zh-en")
        train_dataset = dataset["train"].select(range(min(50000, len(dataset["train"]))))
        validation_dataset = dataset["validation"].select(range(min(1000, len(dataset["validation"]))))
        test_dataset = dataset["test"].select(range(min(1000, len(dataset["test"]))))

        print(f"Train size: {len(train_dataset)}")
        print(f"Validation size: {len(validation_dataset)}")
        print(f"Test size: {len(test_dataset)}")

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
    # 处理不同数据集的格式
    if "translation" in examples:
        # wmt19, ted_talks_iwslt格式
        inputs = [prefix + ex["zh"] for ex in examples["translation"]]
        targets = [ex["en"] for ex in examples["translation"]]
    else:
        # opus100格式
        inputs = [prefix + ex for ex in examples["zh"]]
        targets = [ex for ex in examples["en"]]

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
