from transformers import TrainingArguments
from recipes import (MistralModelRecipe, 
                     MistralTokenizerRecipe, 
                     YAMLDatasetRecipe, 
                     MistralModelTemplateRecipe)
from recipes.QuantizationRecipe import QuantizationRecipe
from recipes.PeftRecipe import PeftRecipe
from configs.Config import Config


config = Config(
    dataset_name="math_qa",
    dataset_recipe=YAMLDatasetRecipe(yaml_path="../lm-evaluation-harness-prompt-template/lm_eval/tasks/mathqa/mathqa.yaml"),
    num_examples=1,
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    model_recipe=MistralModelRecipe(
        model_load={
            "cache_dir": "../models", 
            "use_cache": True, 
            "max_position_embeddings": 512
        }
    ),
    model_template_recipe=MistralModelTemplateRecipe(),
    tokenizer_recipe=MistralTokenizerRecipe(
        tokenizer_config={
            "cache_dir": "../models", 
            "model_max_length": 512
        }
    ),
    training_arguments=TrainingArguments(
        output_dir="../models/logiqa/mistral_instruct/finetune",
        num_train_epochs=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=3e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        tf32=False,
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
    ),
    finetuner_arguments={
        "max_seq_length": 512
    },
    completion_only=True
)
