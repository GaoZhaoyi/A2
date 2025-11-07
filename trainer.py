from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> TrainingArguments:
    """
    Create and return the training arguments for the model.

    Returns:
        Training arguments for the model.

    NOTE: You can change the training arguments as needed.
    # Below is an example of how to create training arguments. You are free to change this.
    # ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        max_grad_norm=1.0,
        predict_with_generate=True,
        fp16=True,
        gradient_accumulation_steps=1,
        dataloader_num_workers=2,
        generation_max_length=False,
        generation_num_beams=4,
        report_to="none",  # Disable wandb to avoid login issues
        # 添加时间优化参数
        dataloader_prefetch_factor=2,
        remove_unused_columns=True,
    )

    return training_args


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks.

    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        DataCollatorForSeq2Seq instance.

    NOTE: You are free to change this. But make sure the data collator is the same as the model.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def build_trainer(model, tokenizer, tokenized_datasets) -> Trainer:
    """
    Build and return the trainer object for training and evaluation.

    Args:
        model: Model for sequence-to-sequence tasks.
        tokenizer: Tokenizer object.
        tokenized_datasets: Tokenized datasets.

    Returns:
        Trainer object for training and evaluation.

    NOTE: You are free to change this. But make sure the trainer is the same as the model.
    """
    # 创建数据整理器，用于将多个样本组合成批次
    data_collator = create_data_collator(tokenizer, model)
    # 获取训练参数配置
    training_args: TrainingArguments = create_training_arguments()

    # 忽略tokenizer警告
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # 创建Seq2SeqTrainer对象，这是Hugging Face提供的专门用于序列到序列任务的训练器
    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )
