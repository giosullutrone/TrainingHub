from transformers import TrainingArguments
from recipes import DatasetRecipe, ModelRecipe, MistralModelRecipe, TokenizerRecipe, MistralTokenizerRecipe, YAMLDatasetRecipe
from recipes.QuantizationRecipe import QuantizationRecipe
from recipes.PeftRecipe import PeftRecipe
from configs.Config import Config

config = Config(
    dataset_name=""
)

training_arguments: TrainingArguments = TrainingArguments(
    output_dir="../models/mistral_instruct/finetune",
    num_train_epochs=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    learning_rate=3e-4,
    weight_decay=0.001,
    fp16=True,
    tf32=True,
    max_grad_norm=0.3,
    max_steps=1000,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    logging_steps=0.25,
    eval_steps=0.25,
    save_steps=0.25,
    load_best_model_at_end=True,
    save_total_limit=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2
)

tokenizer_recipe: TokenizerRecipe = MistralTokenizerRecipe
tokenizer_arguments = {
    "cache_dir": "../models", 
    "model_max_length": 512
}

model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
model_recipe: ModelRecipe = MistralModelRecipe
model_arguments = {
    "cache_dir": "../models", 
    "use_cache": True, 
    "max_position_embeddings": 512
}

quantization_recipe: QuantizationRecipe = None
peft_recipe: PeftRecipe = None

dataset_name: str = ""
dataset_recipe: DatasetRecipe = YAMLDatasetRecipe
validation_split_size: float = 0.15
dataset_arguments = {
    "n_example": 0,
    "cache_dir": "../datasets"
}

finetuner_arguments = {
    "max_seq_length": 512
}