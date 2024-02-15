import os
import subprocess
from recipes.TokenizerRecipe import TokenizerRecipe, MistralTokenizerRecipe
from dispatchers.TokenizerDispatcher import TokenizerDispatcher
from recipes.ModelRecipe import MistralModelRecipe, ModelRecipe
from recipes import QuantizationRecipe, MistralQuantizationRecipe, MistralModelTemplateRecipe, YAMLDatasetRecipe
from dispatchers import QuantizationDispatcher, ModelDispatcher, DatasetDispatcher
from recipes.DatasetRecipe import get_examples_lm_evaluation_harness_format

dataset_map = {}
dataset_map["algebra"]="../lm-evaluation-harness-prompt-template/lm_eval/tasks/minerva_math_zero/minerva_math_algebra.yaml"
dataset_map["counting_and_probability"]="../lm-evaluation-harness-prompt-template/lm_eval/tasks/minerva_math_zero/minerva_math_counting_and_prob.yaml"
dataset_map["geometry"]="../lm-evaluation-harness-prompt-template/lm_eval/tasks/minerva_math_zero/minerva_math_geometry.yaml"
dataset_map["intermediate_algebra"]="../lm-evaluation-harness-prompt-template/lm_eval/tasks/minerva_math_zero/minerva_math_intermediate_algebra.yaml"
dataset_map["number_theory"]="../lm-evaluation-harness-prompt-template/lm_eval/tasks/minerva_math_zero/minerva_math_num_theory.yaml"
dataset_map["prealgebra"]="../lm-evaluation-harness-prompt-template/lm_eval/tasks/minerva_math_zero/minerva_math_prealgebra.yaml"
dataset_map["precalculus"]="../lm-evaluation-harness-prompt-template/lm_eval/tasks/minerva_math_zero/minerva_math_precalc.yaml"

model_names = [
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "OpenPipe/mistral-ft-optimized-1227",
    "../models/mistral_base/{dataset}/finetune",
    "../models/mistral_instruct/{dataset}/finetune",
    "../models/mistral_opt/{dataset}/finetune",
    "../models/mistral_base/{dataset}/finetune_limited",
    "../models/mistral_instruct/{dataset}/finetune_limited",
    "../models/mistral_opt/{dataset}/finetune_limited",
]
tokenizer_names = [
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "OpenPipe/mistral-ft-optimized-1227",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "OpenPipe/mistral-ft-optimized-1227",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "OpenPipe/mistral-ft-optimized-1227",
]

for model_name, tokenizer_name in zip(
    model_names,
    tokenizer_names
):
    tokenizer_recipe: TokenizerRecipe = MistralTokenizerRecipe()
    tokenizer = TokenizerDispatcher(tokenizer_recipe).get_tokenizer(tokenizer_name)
    model_template_recipe = MistralModelTemplateRecipe()

    if "models" not in model_name: continue # To Remove
    is_local = False

    for dataset in dataset_map:
        yaml_path: str = dataset_map[dataset]
        dataset_name = "EleutherAI/hendrycks_math"
        task_name = yaml_path.split("/")[-1].replace(".yaml", "")
        _model_name = model_name.format_map({"dataset": dataset})
        if os.path.exists(_model_name):
            output_path = os.path.join(_model_name, f"results.txt")
            _model_name = [os.path.join(_model_name, item) for item in os.listdir(_model_name) if os.path.isdir(os.path.join(_model_name, item))][0]
            is_local = True
        else: output_path = f"../models/{model_name.split('/')[-1]}/{dataset}/results.txt"

        dataset_support, dataset_train = DatasetDispatcher(YAMLDatasetRecipe(dataset_load={"yaml_path": yaml_path, 
                                                                                           "name": dataset})).get_support_and_tuning_dataset(
                                                                                                            dataset_name, 
                                                                                                            split="test", 
                                                                                                            num_examples=0,
                                                                                                            postprocess_function=model_template_recipe.postprocess_function,
                                                                                                            eos_token=tokenizer.eos_token,
                                                                                                            include_labels_inside_text=True)

        prompt = get_examples_lm_evaluation_harness_format(dataset_support)
        prompt = f"[INST] \n{prompt}" + "{text} [/INST]"
        prompt = "{text}"
        print(prompt)
        if is_local:
            comm = f'python /mnt/e/Studio/Dottorato/lm-evaluation-harness-prompt-template/lm_eval --model hf --model_args pretrained="{tokenizer_name}",peft="{_model_name}",load_in_4bit=True,bnb_4bit_quant_type=nf4 --tasks {task_name} --num_fewshot 0 --verbosity DEBUG --text_to_prompt "{prompt}"'
        else:
            comm = f'python /mnt/e/Studio/Dottorato/lm-evaluation-harness-prompt-template/lm_eval --model hf --model_args pretrained="{_model_name}",load_in_4bit=True,bnb_4bit_quant_type=nf4 --tasks {task_name} --num_fewshot 0 --verbosity DEBUG --text_to_prompt "{prompt}"'
        os.makedirs(output_path.replace("results.txt", ""), exist_ok=True)
        with open(output_path, "w") as f:
            subprocess.run(comm, shell=True, stdout=f)
