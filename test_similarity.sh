#!/bin/bash

python similarity_cli.py -c configs/similarity_yaml.py \
--output_path "../models/mistral/mathqa/result.json" \
--dataset_name "math_qa" \
--dataset_recipe "YAMLDatasetRecipe" \
--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/mathqa/mathqa.yaml"}' \
--num_examples 3 \
--model_name "mistralai/Mistral-7B-Instruct-v0.1" \
--tokenizer_name "mistralai/Mistral-7B-Instruct-v0.1" \
--model_recipe "MistralModelRecipe" \
--model_template_recipe "MistralModelTemplateRecipe" \
--model_load '{"cache_dir": "../models"}' \
--tokenizer_recipe "MistralTokenizerRecipe" \
--tokenizer_load '{"cache_dir": "../models"}' \
--model_finetune_path "../models/mistral/mathqa/finetune_3shot/checkpoint-3357" \
--model_peft_path "../models/mistral/mathqa/spt_3shot/checkpoint-3357" \
--num_samples 250

python similarity_cli.py -c configs/similarity_yaml.py \
--output_path "../models/mistral_base/mathqa/result.json" \
--dataset_name "math_qa" \
--dataset_recipe "YAMLDatasetRecipe" \
--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/mathqa/mathqa.yaml"}' \
--num_examples 3 \
--model_name "mistralai/Mistral-7B-v0.1" \
--tokenizer_name "mistralai/Mistral-7B-v0.1" \
--model_recipe "MistralModelRecipe" \
--model_template_recipe "MistralModelTemplateRecipe" \
--model_load '{"cache_dir": "../models"}' \
--tokenizer_recipe "MistralTokenizerRecipe" \
--tokenizer_load '{"cache_dir": "../models"}' \
--model_finetune_path "../models/mistral_base/mathqa/finetune_3shot/checkpoint-3357" \
--model_peft_path "../models/mistral_base/mathqa/spt_3shot/checkpoint-3357" \
--num_samples 250

python similarity_cli.py -c configs/similarity_yaml.py \
--output_path "../models/mistral_opt/mathqa/result.json" \
--dataset_name "math_qa" \
--dataset_recipe "YAMLDatasetRecipe" \
--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/mathqa/mathqa.yaml"}' \
--num_examples 3 \
--model_name "OpenPipe/mistral-ft-optimized-1227" \
--tokenizer_name "OpenPipe/mistral-ft-optimized-1227" \
--model_recipe "MistralModelRecipe" \
--model_template_recipe "MistralModelTemplateRecipe" \
--model_load '{"cache_dir": "../models"}' \
--tokenizer_recipe "MistralTokenizerRecipe" \
--tokenizer_load '{"cache_dir": "../models"}' \
--model_finetune_path "../models/mistral_opt/mathqa/finetune_3shot/checkpoint-3357" \
--model_peft_path "../models/mistral_opt/mathqa/spt_3shot/checkpoint-3357" \
--num_samples 250