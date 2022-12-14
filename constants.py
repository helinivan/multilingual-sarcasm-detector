# Values
SEED = 42
MAX_LENGTH = 256
TFID_MAX_FEATURES_MONOLINGUAL = 30000
TFID_MAX_FEATURES_MULTILINGUAL = 90000
NUMBER_OF_TRIALS = 10
BERT_MODEL_OPTIMIZATION_METRIC = "eval_loss"

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
