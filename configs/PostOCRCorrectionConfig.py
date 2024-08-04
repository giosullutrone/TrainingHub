from recipes.quantizations.FourBitQuantizationRecipe import FourBitQuantizationRecipe
from recipes.models.MistralModelRecipe import MistralModelRecipe
from recipes.tokenizers.MistralTokenizerRecipe import MistralTokenizerRecipe
from recipes.pefts.LoRaPeftRecipe import LoRaPeftRecipe
from recipes.datasets.PostOCRCorrectionDatasetRecipe import PostOCRCorrectionDatasetRecipe
from recipes.train import DefaultTrainRecipe
from configs import Config
from datasets import DownloadMode


config = Config( 
    dataset_recipe=PostOCRCorrectionDatasetRecipe(
        dataset_load={"name": "english", "download_mode": DownloadMode.REUSE_DATASET_IF_EXISTS, "cache_dir": "E:\\Studio\\Dottorato\\dataset"},
        dataset_system_message="You are a bot specialied in OCR correction, respond to the user with the corrected OCR text." 
    ),
    model_recipe=MistralModelRecipe(
        model_load={
            **MistralModelRecipe.model_load,
            "cache_dir": "../../models",
            "use_cache": True,
        },
    ),
    tokenizer_recipe=MistralTokenizerRecipe(
        tokenizer_config={
            **MistralTokenizerRecipe.tokenizer_config,
            "cache_dir": "../models",
            "model_max_length": 2048,
        },
    ),
    peft_recipe=LoRaPeftRecipe(),
    quantization_recipe=FourBitQuantizationRecipe(),
    train_recipe=DefaultTrainRecipe(
        dataset_name="PleIAs/Post-OCR-Correction",
        training_split="train[:80%]", 
        validation_split="train[80%:]",
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
        training_arguments={
            "num_train_epochs": 1,
            "gradient_accumulation_steps": 1,
            "optim": "adamw_torch",
            "learning_rate": 1e-5,
            "weight_decay": 0.001,
            "fp16": False,
            "bf16": True,
            "tf32": False,
            "max_grad_norm": 0.3,
            "max_steps": -1,
            "warmup_ratio": 0.03,
            "group_by_length": True,
            "lr_scheduler_type": "cosine",
            "eval_strategy": "steps",
            "logging_steps": 0.25,
            "eval_steps": 0.25,
            "save_steps": 0.25,
            "load_best_model_at_end": False,
            "save_total_limit": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 2,
            "logging_first_step": True,
            "output_dir": "E:\\Studio\\Dottorato\\models",
            "max_seq_length": 2048,
            "packing": False
        },
        completion_only=True,
        num_proc=12,
        shuffle=True,
        shuffle_seed=42,
        dataset_size=1000
    )
)
