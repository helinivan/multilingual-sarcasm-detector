""""

Data Preprocessing step inspired by: https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

"""


import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")

import pandas as pd
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import os

# Current path
CURRENT_DIRECTORY = os.getcwd()


# Pre-process dataset
def data_preprocessing(dataset: pd.DataFrame = None) -> pd.DataFrame:
    """Function pre-processes text data using tokenizing and lemmatization
  Args:
    dataset (pd.DataFrame): Dataset containing all the data
  Returns:
    dataset (pd.DataFrame): Dataset containing preprocessed text data
  """

    def preprocess_text(text: str) -> str:

        # Declaring empty list to store the words that follow the rules for this step
        preprocessed_data = []

        # post_tag function below will provide the "tag" i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(text):

            # Below condition is to check for stopwords words and consider only alphabets

            if (
                (word not in stopwords.words("english"))
                and (word not in stopwords.words("italian"))
                and (word not in stopwords.words("dutch"))
                and (word.isalpha())
            ):

                word_processed = word_lemmatizer.lemmatize(word, tag_map[tag[0]])
                preprocessed_data.append(word_processed)

        return str(preprocessed_data)

    # If input dataset not specified, load dataset from training_data folder
    if not dataset:
        dataset = pd.read_csv(
            f"{CURRENT_DIRECTORY}/training_data/multilang_sarcasm_dataset.csv"
        )

    # Lowercase
    dataset["article_title_preprocessed"] = [
        entry.lower() for entry in dataset["article_title"]
    ]

    # Tokenize text
    dataset["article_title_preprocessed"] = [
        word_tokenize(entry) for entry in dataset["article_title_preprocessed"]
    ]

    # Remove stopwords, non-numeric and perfom lemmatization
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map["J"] = wn.ADJ
    tag_map["V"] = wn.VERB
    tag_map["R"] = wn.ADV

    # Initializing WordNetLemmatizer()
    word_lemmatizer = WordNetLemmatizer()

    # Pre-process each article title using preprocess_text()
    dataset["article_title_preprocessed"] = dataset["article_title_preprocessed"].apply(
        lambda text: preprocess_text(text)
    )

    # Save preprocessed dataset to training_data folder
    dataset.to_csv(
        f"{CURRENT_DIRECTORY}/training_data/multilang_sarcasm_dataset_preprocessed.csv",
        index=False,
    )

    return dataset
