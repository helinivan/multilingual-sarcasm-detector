"""
Hyperparameter tuning inspired by: https://neptune.ai/blog/optuna-guide-how-to-monitor-hyper-parameter-optimization-runs

"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
import optuna
import time
import os
import json
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from constants import (
    SEED,
    TFID_MAX_FEATURES,
    NUMBER_OF_TRIALS,
    LANGUAGES_DICT,
    DUTCH,
    ENGLISH,
    ITALIAN,
    MULTILINGUAL,
    MODEL_LANGS_ENGLISH,
    MODEL_LANGS_IT,
    MODEL_LANGS_NL,
    MODEL_LANGS_ALL,
)

# Set seed
np.random.seed(SEED)

# FILE PATHS
CURRENT_DIRECTORY = cwd = os.getcwd()
TRAINING_DATA_PATH = f"{CURRENT_DIRECTORY}/training_data"
MODEL = "svm"

# Paths to store model
PATHS = {
    "en": f"/{CURRENT_DIRECTORY}/{MODEL}/en-{MODEL}",
    "it": f"/{CURRENT_DIRECTORY}/{MODEL}/it-{MODEL}",
    "nl": f"/{CURRENT_DIRECTORY}/{MODEL}/nl-{MODEL}",
    "it-en": f"/{CURRENT_DIRECTORY}/{MODEL}/m-{MODEL}-it-en",
    "it-nl": f"/{CURRENT_DIRECTORY}/{MODEL}/m-{MODEL}-it-nl",
    "nl-en": f"/{CURRENT_DIRECTORY}/{MODEL}/m-{MODEL}-nl-en",
    "en-it-nl": f"/{CURRENT_DIRECTORY}/{MODEL}/m-{MODEL}-all",
}


class InvalidLanguageInput(Exception):
    pass


# Get output path
def get_output_path(langs: str, model_type: str) -> str:
    """Function defines output path for model files
  Args:
    langs (str): Languages used in model training
    model_type (str): Name of the model
  Returns:
    output_path (str): Output path for model files
  """

    if not langs:
        return None

    if langs == "en" and model_type == "multilingual":
        return f"/{CURRENT_DIRECTORY}/{MODEL}/m-{MODEL}-{langs}"

    if langs == "nl" and model_type == "multilingual":
        return f"/{CURRENT_DIRECTORY}/{MODEL}/m-{MODEL}-{langs}"

    if langs == "it" and model_type == "multilingual":
        return f"/{CURRENT_DIRECTORY}/{MODEL}/m-{MODEL}-{langs}"

    output_path = PATHS[langs] if langs in PATHS else None

    return output_path


# Get training data
def get_training_data(dataset: pd.DataFrame, list_languages: list):
    """Function selects training and testing data based on used langauges
  Args:
    dataset (pd.DataFrame): Dataframe containing all data
    list_languages (list): List of languages used in the training
  Return:
    x (list): Values for training 
    x_val (list): Values for testing
    y (list): Labels for training
    y_val (list): Labels for testing
  """

    # Use selected languages for training and testing
    data_training = dataset[dataset["lang"].isin(list_languages)]

    x = [text for text in data_training["article_title_preprocessed"].values.ravel().astype(str)]
    y = [label for label in data_training["is_sarcastic"].values.ravel().astype(int)]

    # Training and testing data
    x, x_val, y, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )

    return x, x_val, y, y_val, data_training


# Model creator
def create_model(trial):

    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    regularization = trial.suggest_uniform("svm-regularization", 0.01, 10)
    degree = trial.suggest_discrete_uniform("degree", 1, 5, 1)
    model = SVC(kernel=kernel, C=regularization, degree=degree, random_state=SEED)

    return model


# Calculate model performance
def model_performance(model, x_val: str, y_val: int, return_all: bool = False):
    """Get evaluation scores on validation/test data from a trained model
    """

    # Make prediction
    y_pred = model.predict(x_val)

    # Get F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=y_val,
        y_pred=y_pred,
        beta=1.0,
        average="macro",
        warn_for=tuple(),
        zero_division=1,
    )

    # Get accuracy
    accuracy = accuracy_score(y_true=y_val, y_pred=y_pred)

    # Return all evaluation metrics
    if return_all:
        return accuracy, precision, recall, f1

    # Return value that is used to evaluate model performance
    return f1


# Model trainer
def train_model(model_langs: list, model_type: str):
    """Function trains the model based on model type and selected languages
  Args:
    model_langs (list): List of languages used for training
    model_type (str): Type of the model
  """

    # Read csv with training data - dataset has been already preprocessed using data_preprocessing() from preprocess_training_data.py
    data_df = pd.read_csv(
        f"/{TRAINING_DATA_PATH}/multilingual_data_set_preprocessed.csv"
    )
    data_df = data_df.sample(frac=1).reset_index(drop=True)

    # Define languages and model output path
    languages = [
        LANGUAGES_DICT[language_name]
        for language_name in model_langs
        if language_name in LANGUAGES_DICT
    ]
    langs = "-".join(languages) if languages else None
    output_path = get_output_path(langs=langs, model_type=model_type)

    if not langs or not output_path:
        raise InvalidLanguageInput(
            f"No language found from input, input langs: {model_langs}"
        )

    # Get training data and testing data
    x, x_val, y, y_val, data_training = get_training_data(
        dataset=data_df, list_languages=languages,
    )

    # TF-IDF Transform
    tfidf_vect = TfidfVectorizer(max_features=TFID_MAX_FEATURES)
    tfidf_vect.fit(data_training["article_title_preprocessed"])
    x = tfidf_vect.transform(x)
    x_val = tfidf_vect.transform(x_val)

    # Objective function
    def objective(trial) -> float:

        # Fitting model
        model = create_model(trial)
        model.fit(x, y)

        # Evaluate performance
        performance = model_performance(model=model, x_val=x_val, y_val=y_val)

        return performance

    # Train the model using hyperparameter search, maximize for F1 score
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        study_name=f"SVM_{langs}_{model_type}",
    )

    study.optimize(objective, n_trials=NUMBER_OF_TRIALS)

    # Get best model
    best_model = create_model(study.best_trial)
    best_model.fit(x, y)

    # Retrieve best hyperparameters and save
    best_hyperparams = study.best_trial.params
    with open(f".{output_path}/hyperparameters.json", "w") as file_output:
        json.dump(best_hyperparams, file_output)

    # Save best model
    filename = f"svm_{langs}_{model_type}.sav"
    joblib.dump(best_model, f".{output_path}/{filename}")

    # Get all evaluation metrics for best model X_val: str, y_val: int,
    accuracy, precision, recall, f1_score = model_performance(
        model=best_model, x_val=x_val, y_val=y_val, return_all=True
    )

    scores = {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "accuracy": accuracy,
    }

    # Save scores
    with open(f".{output_path}/scores.json", "w") as file_output:
        json.dump(scores, file_output)


if __name__ == "__main__":

    # Measure training time
    start = time.time()

    # 1. Launch training of the English SVM model
    train_model(model_langs=MODEL_LANGS_ENGLISH, model_type=ENGLISH)

    # 2. Launch training of Dutch SVM model
    train_model(model_langs=MODEL_LANGS_NL, model_type=DUTCH)

    # 3. Launch training of Italian SVM model
    train_model(model_langs=MODEL_LANGS_IT, model_type=ITALIAN)

    # 4. Launch training of the multilingual all languages SVM model
    train_model(model_langs=MODEL_LANGS_ALL, model_type=MULTILINGUAL)

    # Print training duration
    print(f"Total training time (s): {time.time() - start}")
