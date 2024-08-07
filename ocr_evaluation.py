from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from dispatchers.QuantizationDispatcher import QuantizationDispatcher
from recipes.quantizations.FourBitQuantizationRecipe import FourBitQuantizationRecipe


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    Args:
    s1 (str): First string.
    s2 (str): Second string.

    Returns:
    int: The Levenshtein distance.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate the Character Error Rate (CER).

    Args:
    reference (str): The correct text.
    hypothesis (str): The text to be evaluated.

    Returns:
    float: The Character Error Rate.
    """
    edit_distance = levenshtein_distance(reference, hypothesis)
    return edit_distance / len(reference)

def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate the Word Error Rate (WER).

    Args:
    reference (str): The correct text.
    hypothesis (str): The text to be evaluated.

    Returns:
    float: The Word Error Rate.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    edit_distance = levenshtein_distance(' '.join(ref_words), ' '.join(hyp_words))
    return edit_distance / len(ref_words)

def correct_ocr_errors(ocr_text: str, chatbot, tokenizer) -> str:
    """
    Attempt to correct OCR errors in the input string using the recurrentgemma model.

    Args:
    ocr_text (str): The OCR-generated text with potential errors.
    model: The recurrentgemma model.
    tokenizer: The tokenizer for the recurrentgemma model.

    Returns:
    str: The corrected text.
    """

    messages = [
        {"role": "system", "content": "You are a bot specialied in OCR correction, respond to the user with the corrected OCR text."},
        {"role": "user", "content": f"Text: {ocr_text}" + "\nCorrected Text:"},
    ]

    corrected_text = chatbot(messages, max_length=2048)[0]['generated_text'][-1]['content'].strip()

    return corrected_text

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")



config = QuantizationDispatcher(FourBitQuantizationRecipe()).get_quantization_config()

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", 
                                             torch_dtype=torch.bfloat16, 
                                             device_map="auto", 
                                             quantization_config=config)
adapter = model.load_adapter("E:\\Studio\\Dottorato\\models\\checkpoint-1000")
model.active_adapters = adapter
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load the dataset
dataset = load_dataset("PleIAs/Post-OCR-Correction", "english", split="train")

# Filter the dataset
filtered_dataset = dataset.filter(lambda example: len(example['text']) < int(1024 * 3))
print(filtered_dataset)

# Process the dataset
# Initialize a list to store all results
results = []

# Process the dataset
for example in tqdm(filtered_dataset, desc="Processing dataset"):
    ocr_text = example['text']
    reference_text = example['corrected_text']

    # Calculate original error rates
    original_cer = character_error_rate(reference_text, ocr_text)
    original_wer = word_error_rate(reference_text, ocr_text)

    # Correct OCR errors
    corrected_text = correct_ocr_errors(ocr_text, chatbot, tokenizer)
    print(corrected_text)

    # Calculate corrected error rates
    corrected_cer = character_error_rate(reference_text, corrected_text)
    corrected_wer = word_error_rate(reference_text, corrected_text)

    # Store the result for this example
    result = {
        "original_text": ocr_text,
        "corrected_text": corrected_text,
        "reference_text": reference_text,
        "original_cer": original_cer,
        "corrected_cer": corrected_cer,
        "original_wer": original_wer,
        "corrected_wer": corrected_wer
    }
    results.append(result)

# Calculate average error rates
avg_original_cer = sum(r["original_cer"] for r in results) / len(results)
avg_original_wer = sum(r["original_wer"] for r in results) / len(results)
avg_corrected_cer = sum(r["corrected_cer"] for r in results) / len(results)
avg_corrected_wer = sum(r["corrected_wer"] for r in results) / len(results)

# Calculate improvement percentages
cer_improvement = (avg_original_cer - avg_corrected_cer) / avg_original_cer * 100
wer_improvement = (avg_original_wer - avg_corrected_wer) / avg_original_wer * 100

# Create a dictionary with all the data
output_data = {
    "results": results,
    "statistics": {
        "avg_original_cer": avg_original_cer,
        "avg_corrected_cer": avg_corrected_cer,
        "avg_original_wer": avg_original_wer,
        "avg_corrected_wer": avg_corrected_wer,
        "cer_improvement": cer_improvement,
        "wer_improvement": wer_improvement
    }
}

# Save the results to a JSON file
with open('ocr_correction_results.json', 'w', encoding='utf-8') as jsonfile:
    json.dump(output_data, jsonfile, ensure_ascii=False, indent=2)

# Print results
print(f"\nAverage Character Error Rate (CER):")
print(f"Before correction: {avg_original_cer:.4f}")
print(f"After correction: {avg_corrected_cer:.4f}")

print(f"\nAverage Word Error Rate (WER):")
print(f"Before correction: {avg_original_wer:.4f}")
print(f"After correction: {avg_corrected_wer:.4f}")

print(f"\nImprovement in CER: {cer_improvement:.2f}%")
print(f"Improvement in WER: {wer_improvement:.2f}%")

print("\nResults have been saved to 'ocr_correction_results.json'")
