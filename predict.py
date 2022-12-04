# Import libraries
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import string

# Preprocess input data
def preprocess_data(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()


HUGGINGFACE_SARCASM_MODELS = [
    "helinivan/multilingual-sarcasm-detector",
    "helinivan/english-sarcasm-detector",
    "helinivan/italian-sarcasm-detector",
    "helinivan/dutch-sarcasm-detector",
]

# Prediction example
MODEL_PATH = "helinivan/multilingual-sarcasm-detector"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

text = "CIA Realizes It's Been Using Black Highlighters All These Years."
tokenized_text = tokenizer(
    [preprocess_data(text)],
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt",
)
output = model(**tokenized_text)
probs = output.logits.softmax(dim=-1).tolist()[0]
confidence = max(probs)
prediction = probs.index(confidence)
results = {"is_sarcastic": prediction, "confidence": confidence}