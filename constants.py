# Values
SEED = 42
MAX_LENGTH = 512
TFID_MAX_FEATURES = 10000
NUMBER_OF_TRIALS = 10
BERT_MODEL_OPTIMIZATION_METRIC = "eval_f1"

# Model types
DUTCH = "dutch"
ENGLISH = "english"
ITALIAN = "italian"
MULTILINGUAL = "multilingual"

# Models
BERT_EN = "bert-base-uncased"
BERT_MULTILANG = "bert-base-multilingual-uncased"
BERT_IT = "dbmdz/bert-base-italian-uncased"
BERT_NL = "GroNLP/bert-base-dutch-cased"

# Languages
MODEL_LANGS_ENGLISH = ["English"]
MODEL_LANGS_IT = ["Italian"]
MODEL_LANGS_NL = ["Dutch"]
MODEL_LANGS_IT_EN = ["Italian", "English"]
MODEL_LANGS_NL_EN = ["Dutch", "English"]
MODEL_LANGS_IT_NL = ["Italian", "Dutch"]
MODEL_LANGS_ALL = ["English", "Italian", "Dutch"]

# Languages dictionary from language name to ISO code
LANGUAGES_DICT = {
    "English": "en",
    "Dutch": "nl",
    "Italian": "it",
}
