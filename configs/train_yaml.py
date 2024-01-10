from configs.Config import Config


config = Config(
    dataset_name="math_qa",
    dataset_recipe="YAMLDatasetRecipe",
    training_arguments={
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 8,
        "optim": "adamw_torch",
        "learning_rate": 1e-5,
        "weight_decay": 0.001,
        "bf16": True,
        "tf32": True,
        "max_grad_norm": 0.3,
        "max_steps": -1,
        "warmup_ratio": 0.03,
        "group_by_length": True,
        "lr_scheduler_type": "cosine",
        "evaluation_strategy": "steps",
        "logging_steps": 0.1,
        "eval_steps": 0.1,
        "save_steps": 0.1,
        "load_best_model_at_end": False,
        "save_total_limit": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 2,
    },
    finetuner_arguments={
        "max_seq_length": 512
    },
    completion_only=True
)