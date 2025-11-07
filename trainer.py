from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import warnings

from constants import OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> Seq2SeqTrainingArguments:
    """
    Create training arguments optimized for high BLEU score within time constraints.
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,  # 增加到3轮以提高性能
        per_device_train_batch_size=32,  # 适中的batch size
        per_device_eval_batch_size=32,
        learning_rate=2e-4,  # 优化的学习率
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=50,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        max_grad_norm=1.0,
        predict_with_generate=True,
        fp16=True,
        gradient_accumulation_steps=2,  # 梯度累积模拟更大的batch
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        report_to="none",
        generation_max_length=128,
        generation_num_beams=3,  # 减少beam search以节省时间
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
    )

    return training_args


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def build_trainer(model, tokenizer, tokenized_datasets) -> Seq2SeqTrainer:
    """
    Build trainer optimized for high BLEU score.
    """
    data_collator = create_data_collator(tokenizer, model)
    training_args = create_training_arguments()

    # 忽略特定警告
    warnings.filterwarnings("ignore", category=FutureWarning)

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )
