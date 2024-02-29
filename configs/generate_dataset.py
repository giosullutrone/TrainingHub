from configs import ConfigGenerateDataset
from transformers import GenerationConfig


config = ConfigGenerateDataset(
    generation_config=GenerationConfig(
        num_return_sequences=4,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        max_new_tokens=300
    ),
    generation_num_examples=1
)