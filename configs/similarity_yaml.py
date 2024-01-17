from configs import ConfigSimilarity


config = ConfigSimilarity(
    num_examples=3,
    tokenizer_config={
        "padding_side": "left", 
        "from_slow":  True
    },
    system_tuning=True
)