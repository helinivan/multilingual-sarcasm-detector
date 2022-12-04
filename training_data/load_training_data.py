# Import libraries
from datasets import load_dataset
import os

# FILE PATHS
CURRENT_DIRECTORY = os.getcwd()

# Load dataset from HuggingFace repository
def load_dataset_from_hf() -> str:

    # Load dataset from HuggingFace
    multilang_sarcasm_dataset = (
        load_dataset(
            "helinivan/sarcasm_headlines_multilingual", 
            data_files={"train": "multilang_sarcasm_dataset.csv"},
            split="train")
            .to_pandas()
        )

    # Save dataset
    multilang_sarcasm_dataset.to_csv(f"{CURRENT_DIRECTORY}/training_data/multilang_sarcasm_dataset.csv", index=False)

    return 'Success'
