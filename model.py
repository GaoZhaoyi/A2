from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel, \
    AutoModelForSeq2SeqLM

from constants import MODEL_CHECKPOINT


def initialize_tokenizer() -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """
    Initialize a tokenizer for sequence-to-sequence tasks.

    Returns:
        A tokenizer for sequence-to-sequence tasks.

    NOTE: You are free to change this. But make sure the tokenizer is the same as the model.
    """
    # 根据模型名称加载对应的分词器
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT
    )

    # 对于M2M100这样的多语言模型，需要指定源语言和目标语言
    if "m2m100" in MODEL_CHECKPOINT.lower():
        tokenizer.src_lang = "zh"
        tokenizer.tgt_lang = "en"

    return tokenizer


def initialize_model() -> PreTrainedModel:
    """
    Initialize a model for sequence-to-sequence tasks. You are free to change this,
    not only seq2seq models, but also other models like BERT, or even LLMs.

    Returns:
        A model for sequence-to-sequence tasks.

    NOTE: You are free to change this.
    """
    # 加载序列到序列的预训练模型
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT
    )

    # 对于M2M100模型，设置生成时必须以特定token开始
    if "m2m100" in MODEL_CHECKPOINT.lower():
        model.config.forced_bos_token_id = model.config.decoder_start_token_id

    return model
