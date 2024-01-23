#!/bin/bash

#python similarity_cli.py -c configs/similarity_yaml.py \
#--output_path "../models/mistral/mathqa/result.json" \
#--dataset_name "math_qa" \
#--dataset_recipe "YAMLDatasetRecipe" \
#--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/mathqa/mathqa.yaml"}' \
#--num_examples 3 \
#--model_name "mistralai/Mistral-7B-Instruct-v0.1" \
#--tokenizer_name "mistralai/Mistral-7B-Instruct-v0.1" \
#--model_recipe "MistralModelRecipe" \
#--model_template_recipe "MistralModelTemplateRecipe" \
#--model_load '{"cache_dir": "../models"}' \
#--tokenizer_recipe "MistralTokenizerRecipe" \
#--tokenizer_load '{"cache_dir": "../models"}' \
#--model_finetune_path "../models/mistral/mathqa/finetune_3shot/checkpoint-3357" \
#--model_peft_path "../models/mistral/mathqa/spt_3shot/checkpoint-3357" \
#--num_samples 250 \
#--system_tuning True
#
#python similarity_cli.py -c configs/similarity_yaml.py \
#--output_path "../models/mistral_base/mathqa/result.json" \
#--dataset_name "math_qa" \
#--dataset_recipe "YAMLDatasetRecipe" \
#--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/mathqa/mathqa.yaml"}' \
#--num_examples 3 \
#--model_name "mistralai/Mistral-7B-v0.1" \
#--tokenizer_name "mistralai/Mistral-7B-v0.1" \
#--model_recipe "MistralModelRecipe" \
#--model_template_recipe "MistralModelTemplateRecipe" \
#--model_load '{"cache_dir": "../models"}' \
#--tokenizer_recipe "MistralTokenizerRecipe" \
#--tokenizer_load '{"cache_dir": "../models"}' \
#--model_finetune_path "../models/mistral_base/mathqa/finetune_3shot/checkpoint-3357" \
#--model_peft_path "../models/mistral_base/mathqa/spt_3shot/checkpoint-3357" \
#--num_samples 250 \
#--system_tuning True
#
#python similarity_cli.py -c configs/similarity_yaml.py \
#--output_path "../models/mistral_opt/mathqa/result.json" \
#--dataset_name "math_qa" \
#--dataset_recipe "YAMLDatasetRecipe" \
#--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/mathqa/mathqa.yaml"}' \
#--num_examples 3 \
#--model_name "OpenPipe/mistral-ft-optimized-1227" \
#--tokenizer_name "OpenPipe/mistral-ft-optimized-1227" \
#--model_recipe "MistralModelRecipe" \
#--model_template_recipe "MistralModelTemplateRecipe" \
#--model_load '{"cache_dir": "../models"}' \
#--tokenizer_recipe "MistralTokenizerRecipe" \
#--tokenizer_load '{"cache_dir": "../models"}' \
#--model_finetune_path "../models/mistral_opt/mathqa/finetune_3shot/checkpoint-3357" \
#--model_peft_path "../models/mistral_opt/mathqa/spt_3shot/checkpoint-3357" \
#--num_samples 250 \
#--system_tuning True

####################################

#python similarity_cli.py -c configs/similarity_yaml.py \
#--output_path "../models/mistral/logiqa/result.json" \
#--dataset_name "EleutherAI/logiqa" \
#--dataset_recipe "YAMLDatasetRecipe" \
#--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/logiqa/logiqa.yaml"}' \
#--num_examples 3 \
#--model_name "mistralai/Mistral-7B-Instruct-v0.1" \
#--tokenizer_name "mistralai/Mistral-7B-Instruct-v0.1" \
#--model_recipe "MistralModelRecipe" --model_load '{"cache_dir": "../models", "max_position_embeddings": 1536}' \
#--model_template_recipe "MistralModelTemplateRecipe" \
#--tokenizer_recipe "MistralTokenizerRecipe" --tokenizer_load '{"cache_dir": "../models", "model_max_length": 1536}' \
#--model_finetune_path "../models/mistral/logiqa/finetune_3shot/checkpoint-3357" \
#--model_peft_path "../models/mistral/logiqa/spt_3shot/checkpoint-3357" \
#--num_samples 250 \
#--system_tuning True
#
#python similarity_cli.py -c configs/similarity_yaml.py \
#--output_path "../models/mistral_base/logiqa/result.json" \
#--dataset_name "EleutherAI/logiqa" \
#--dataset_recipe "YAMLDatasetRecipe" \
#--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/logiqa/logiqa.yaml"}' \
#--num_examples 3 \
#--model_name "mistralai/Mistral-7B-v0.1" \
#--tokenizer_name "mistralai/Mistral-7B-v0.1" \
#--model_recipe "MistralModelRecipe" --model_load '{"cache_dir": "../models", "max_position_embeddings": 1536}' \
#--model_template_recipe "MistralModelTemplateRecipe" \
#--tokenizer_recipe "MistralTokenizerRecipe" --tokenizer_load '{"cache_dir": "../models", "model_max_length": 1536}' \
#--model_finetune_path "../models/mistral_base/logiqa/finetune_3shot/checkpoint-3357" \
#--model_peft_path "../models/mistral_base/logiqa/spt_3shot/checkpoint-3357" \
#--num_samples 250 \
#--system_tuning True
#
#python similarity_cli.py -c configs/similarity_yaml.py \
#--output_path "../models/mistral_opt/logiqa/result.json" \
#--dataset_name "EleutherAI/logiqa" \
#--dataset_recipe "YAMLDatasetRecipe" \
#--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/logiqa/logiqa.yaml"}' \
#--num_examples 3 \
#--model_name "OpenPipe/mistral-ft-optimized-1227" \
#--tokenizer_name "OpenPipe/mistral-ft-optimized-1227" \
#--model_recipe "MistralModelRecipe" --model_load '{"cache_dir": "../models", "max_position_embeddings": 1536}' \
#--model_template_recipe "MistralModelTemplateRecipe" \
#--tokenizer_recipe "MistralTokenizerRecipe" --tokenizer_load '{"cache_dir": "../models", "model_max_length": 1536}' \
#--model_finetune_path "../models/mistral_opt/logiqa/finetune_3shot/checkpoint-3357" \
#--model_peft_path "../models/mistral_opt/logiqa/spt_3shot/checkpoint-3357" \
#--num_samples 250 \
#--system_tuning True

####################################

#python similarity_cli.py -c configs/similarity_yaml.py \
#--output_path "../models/mistral/piqa/result.json" \
#--dataset_name "piqa" \
#--dataset_recipe "YAMLDatasetRecipe" \
#--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/piqa/piqa.yaml"}' \
#--num_examples 3 \
#--model_name "mistralai/Mistral-7B-Instruct-v0.1" \
#--tokenizer_name "mistralai/Mistral-7B-Instruct-v0.1" \
#--model_recipe "MistralModelRecipe" --model_load '{"cache_dir": "../models", "max_position_embeddings": 1536}' \
#--model_template_recipe "MistralModelTemplateRecipe" \
#--tokenizer_recipe "MistralTokenizerRecipe" --tokenizer_load '{"cache_dir": "../models", "model_max_length": 1536}' \
#--model_finetune_path "../models/mistral/piqa/finetune_3shot/checkpoint-1818" \
#--model_peft_path "../models/mistral/piqa/spt_3shot/checkpoint-1818" \
#--num_samples 250 \
#--system_tuning True
#
#python similarity_cli.py -c configs/similarity_yaml.py \
#--output_path "../models/mistral_base/piqa/result.json" \
#--dataset_name "piqa" \
#--dataset_recipe "YAMLDatasetRecipe" \
#--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/piqa/piqa.yaml"}' \
#--num_examples 3 \
#--model_name "mistralai/Mistral-7B-v0.1" \
#--tokenizer_name "mistralai/Mistral-7B-v0.1" \
#--model_recipe "MistralModelRecipe" --model_load '{"cache_dir": "../models", "max_position_embeddings": 1536}' \
#--model_template_recipe "MistralModelTemplateRecipe" \
#--tokenizer_recipe "MistralTokenizerRecipe" --tokenizer_load '{"cache_dir": "../models", "model_max_length": 1536}' \
#--model_finetune_path "../models/mistral_base/piqa/finetune_3shot/checkpoint-1818" \
#--model_peft_path "../models/mistral_base/piqa/spt_3shot/checkpoint-1818" \
#--num_samples 250 \
#--system_tuning True
#
#python similarity_cli.py -c configs/similarity_yaml.py \
#--output_path "../models/mistral_opt/piqa/result.json" \
#--dataset_name "piqa" \
#--dataset_recipe "YAMLDatasetRecipe" \
#--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/piqa/piqa.yaml"}' \
#--num_examples 3 \
#--model_name "OpenPipe/mistral-ft-optimized-1227" \
#--tokenizer_name "OpenPipe/mistral-ft-optimized-1227" \
#--model_recipe "MistralModelRecipe" --model_load '{"cache_dir": "../models", "max_position_embeddings": 1536}' \
#--model_template_recipe "MistralModelTemplateRecipe" \
#--tokenizer_recipe "MistralTokenizerRecipe" --tokenizer_load '{"cache_dir": "../models", "model_max_length": 1536}' \
#--model_finetune_path "../models/mistral_opt/piqa/finetune_3shot/checkpoint-1818" \
#--model_peft_path "../models/mistral_opt/piqa/spt_3shot/checkpoint-1818" \
#--num_samples 250 \
#--system_tuning True

####################################

python similarity_cli.py -c configs/similarity_yaml.py \
--output_path "../models/mistral/winogrande/result.json" \
--dataset_name "winogrande" \
--dataset_recipe "YAMLDatasetRecipe" \
--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/winogrande/default.yaml", "name": "winogrande_xl"}' \
--num_examples 3 \
--model_name "mistralai/Mistral-7B-Instruct-v0.1" \
--tokenizer_name "mistralai/Mistral-7B-Instruct-v0.1" \
--model_recipe "MistralModelRecipe" --model_load '{"cache_dir": "../models", "max_position_embeddings": 1536}' \
--model_template_recipe "MistralModelTemplateRecipe" \
--tokenizer_recipe "MistralTokenizerRecipe" --tokenizer_load '{"cache_dir": "../models", "model_max_length": 1536}' \
--model_finetune_path "../models/mistral/winogrande/finetune_3shot/checkpoint-4545" \
--model_peft_path "../models/mistral/winogrande/spt_3shot/checkpoint-4545" \
--num_samples 250 \
--system_tuning True \
--split "validation"

python similarity_cli.py -c configs/similarity_yaml.py \
--output_path "../models/mistral_base/winogrande/result.json" \
--dataset_name "winogrande" \
--dataset_recipe "YAMLDatasetRecipe" \
--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/winogrande/default.yaml", "name": "winogrande_xl"}' \
--num_examples 3 \
--model_name "mistralai/Mistral-7B-v0.1" \
--tokenizer_name "mistralai/Mistral-7B-v0.1" \
--model_recipe "MistralModelRecipe" --model_load '{"cache_dir": "../models", "max_position_embeddings": 1536}' \
--model_template_recipe "MistralModelTemplateRecipe" \
--tokenizer_recipe "MistralTokenizerRecipe" --tokenizer_load '{"cache_dir": "../models", "model_max_length": 1536}' \
--model_finetune_path "../models/mistral_base/winogrande/finetune_3shot/checkpoint-4545" \
--model_peft_path "../models/mistral_base/winogrande/spt_3shot/checkpoint-4545" \
--num_samples 250 \
--system_tuning True \
--split "validation"

python similarity_cli.py -c configs/similarity_yaml.py \
--output_path "../models/mistral_opt/winogrande/result.json" \
--dataset_name "winogrande" \
--dataset_recipe "YAMLDatasetRecipe" \
--dataset_load '{"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/winogrande/default.yaml", "name": "winogrande_xl"}' \
--num_examples 3 \
--model_name "OpenPipe/mistral-ft-optimized-1227" \
--tokenizer_name "OpenPipe/mistral-ft-optimized-1227" \
--model_recipe "MistralModelRecipe" --model_load '{"cache_dir": "../models", "max_position_embeddings": 1536}' \
--model_template_recipe "MistralModelTemplateRecipe" \
--tokenizer_recipe "MistralTokenizerRecipe" --tokenizer_load '{"cache_dir": "../models", "model_max_length": 1536}' \
--model_finetune_path "../models/mistral_opt/winogrande/finetune_3shot/checkpoint-4545" \
--model_peft_path "../models/mistral_opt/winogrande/spt_3shot/checkpoint-4545" \
--num_samples 250 \
--system_tuning True \
--split "validation"