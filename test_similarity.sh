#!/bin/bash

/app/eval_llm/bin/python3 similarity_cli.py -c configs/similarity_yaml.py \
--output_path "../models/mistral/mathqa/result.json" \
--dataset_name "math_qa" \
--dataset_recipe "YAMLDatasetRecipe" \
--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/mathqa/mathqa.yaml"}' \
--num_examples 3 \
--model_name "mistralai/Mistral-7B-Instruct-v0.1" \
--tokenizer_name "mistralai/Mistral-7B-Instruct-v0.1" \
--model_recipe "MistralModelRecipe" --model_load '{"cache_dir": "../models"}' \
--tokenizer_recipe "MistralTokenizerRecipe" --tokenizer_load '{"cache_dir": "../models"}' \
--finetune_path "../models/mistral/mathqa/finetune_3shot/checkpoint-3357" \
--peft_path "../models/mistral/mathqa/spt_3shot/checkpoint-3357"