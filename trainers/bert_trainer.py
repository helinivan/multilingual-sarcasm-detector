import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
import torch
import optuna
from transformers import (
    TrainingArguments,
    Trainer,
    BertTokenizer,
    BertForSequenceClassification,
    EarlyStoppingCallback,
)
import time
import os
import json

from constants import (
    SEED,
    MAX_LENGTH,
    NUMBER_OF_TRIALS,
    BERT_MODEL_OPTIMIZATION_METRIC,
    LANGUAGES_DICT,
    BERT_EN,
    BERT_MULTILANG,
    BERT_IT,
    BERT_NL,
    MODEL_LANGS_ENGLISH,
    MODEL_LANGS_IT,
    MODEL_LANGS_NL,
    MODEL_LANGS_IT_EN,
    MODEL_LANGS_NL_EN,
    MODEL_LANGS_IT_NL,
    MODEL_LANGS_ALL,
)


# SET SEEDS
np.random.seed(SEED)
torch.manual_seed(SEED)

# FILE PATHS
CURRENT_DIRECTORY = os.getcwd()
TRAINING_DATA_PATH = f"{CURRENT_DIRECTORY}/training_data"
MODEL = "bert"

# Paths to store model
PATHS = {
    "en": f"/{CURRENT_DIRECTORY}/results/{MODEL}/en-{MODEL}",
    "it": f"/{CURRENT_DIRECTORY}/results/{MODEL}/it-{MODEL}",
    "nl": f"/{CURRENT_DIRECTORY}/results/{MODEL}/nl-{MODEL}",
    "it-en": f"/{CURRENT_DIRECTORY}/results/{MODEL}/m-{MODEL}-it-en",
    "it-nl": f"/{CURRENT_DIRECTORY}/results/{MODEL}/m-{MODEL}-it-nl",
    "nl-en": f"/{CURRENT_DIRECTORY}/results/{MODEL}/m-{MODEL}-nl-en",
    "en-it-nl": f"/{CURRENT_DIRECTORY}/results/{MODEL}/m-{MODEL}-all",
}


class InvalidLanguageInput(Exception):
    pass


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# Compute evaluation metrics
def compute_metrics(pred) -> dict:

    # Get prediction labels
    labels = pred.label_ids
    predictions = pred.predictions.argmax(-1)

    # Get Precision, Recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=labels, y_pred=predictions, beta=1.0, average="macro", zero_division=0,
    )

    # Get accuracy
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)

    scores = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

    return scores


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
        return f"/{CURRENT_DIRECTORY}/results/{MODEL}/m-{MODEL}-{langs}"

    if langs == "nl" and model_type == "multilingual":
        return f"/{CURRENT_DIRECTORY}/results/{MODEL}/m-{MODEL}-{langs}"

    if langs == "it" and model_type == "multilingual":
        return f"/{CURRENT_DIRECTORY}/results/{MODEL}/m-{MODEL}-{langs}"

    output_path = PATHS[langs] if langs in PATHS else None

    return output_path


# Get training data
def get_training_data(
    dataset: pd.DataFrame, list_languages: list, excluded_langs_as_test: bool = False
):
    """Function selects training and testing data based on used langauges
  Args:
    dataset (pd.DataFrame): Dataframe containing all data
    list_languages (list): List of languages used in the training
    excluded_langs_as_test (bool): Whether languages outside of selected languages are 
                                   utilized as test set, default is False
  Return:
    x (list): Values for training 
    x_val (list): Values for testing
    y (list): Labels for training
    y_val (list): Labels for testing
  """

    # If two languages selected for training, use excluded language data as testing data
    if excluded_langs_as_test:

        # Use selected languages for training
        data_training = dataset.loc[dataset["lang"].isin(list_languages)]

        # Use excluded language as the testing data
        data_testing = dataset.loc[~dataset["lang"].isin(list_languages)]

        x_train = [
            text for text in data_training["article_title"].values.ravel().astype(str)
        ]
        y_train = [
            label for label in data_training["is_sarcastic"].values.ravel().astype(int)
        ]

        # Testing data
        x_test = [
            text for text in data_testing["article_title"].values.ravel().astype(str)
        ]
        y_test = [
            label for label in data_testing["is_sarcastic"].values.ravel().astype(int)
        ]

        # Training / validation data split
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.10, stratify=y_train, random_state=SEED
        )

    else:

        # Use selected languages for training and testing
        dataset = dataset[dataset["lang"].isin(list_languages)]

        x = [text for text in dataset["article_title"].values.ravel().astype(str)]
        y = [label for label in dataset["is_sarcastic"].values.ravel().astype(int)]

        # Training / test data split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.10, stratify=y, random_state=SEED
        )

        # Training / validation data split
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.10, stratify=y_train, random_state=SEED
        )

    return x_train, x_val, y_train, y_val, x_test, y_test


# Hyperparameter tuning
def hyperparameter_space(trial: optuna.trial._trial.Trial) -> dict:

    hyperparameters_dict = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32, 64]
        ),
        "weight_decay": trial.suggest_float("weight_decay", 0.1, 0.5, log=True),
        "optim": trial.suggest_categorical("optim", ["adamw_hf", "adafactor"]),
    }

    return hyperparameters_dict


# Model trainer
def train_model(
    model_langs: list, model_type: str, excluded_langs_as_test: bool = False
):
    """Function trains the model based on model type and selected languages
  Args:
    model_langs (list): List of languages used for training
    model_type (str): Name of the model
    excluded_langs_as_test (bool): Whether languages outside of selected languages are 
                                   utilized as test set, default is False
  """
    # Read csv with training data
    data_df = pd.read_csv(f"{TRAINING_DATA_PATH}/multilang_sarcasm_dataset.csv")
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

    # Get training / evaluation / testing data
    x_train, x_val, y_train, y_val, x_test, y_test = get_training_data(
        dataset=data_df,
        list_languages=languages,
        excluded_langs_as_test=excluded_langs_as_test,
    )

    data_lengths = {
        "training_data_size": len(x_train),
        "evaluation_data_size": len(x_val),
        "test_data_size": len(x_test),
    }

    # Save training / evaluation / test set sizes
    with open(f".{output_path}/training_test_data_size.json", "w") as file_output:
        json.dump(data_lengths, file_output)

    # Define pre-trained model
    def model_init():
        return BertForSequenceClassification.from_pretrained(model_type, num_labels=2)

    # Define pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_type)

    # Tokenize training / validation / sets
    x_train_tokenized = tokenizer(
        x_train, padding=True, truncation=True, max_length=MAX_LENGTH
    )
    x_val_tokenized = tokenizer(
        x_val, padding=True, truncation=True, max_length=MAX_LENGTH
    )
    x_test_tokenized = tokenizer(
        x_test, padding=True, truncation=True, max_length=MAX_LENGTH
    )

    train_dataset = Dataset(x_train_tokenized, y_train)
    eval_dataset = Dataset(x_val_tokenized, y_val)
    test_dataset = Dataset(x_test_tokenized, y_test)

    # Define inputs for Training Args
    args_inputs = {
        "output_dir": f".{output_path}/partial_results",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "num_train_epochs": 10,
        "log_level": "info",
        "seed": SEED,
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": BERT_MODEL_OPTIMIZATION_METRIC,
        "logging_dir": f"./{output_path}/logs",
    }

    # Define Training Args
    params = TrainingArguments(**args_inputs)

    # Define inputs for Trainer
    training_inputs = {
        "model_init": model_init,
        "args": params,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
    }

    # Define Trainer
    trainer = Trainer(**training_inputs)

    def my_objective(metrics):
        return metrics[BERT_MODEL_OPTIMIZATION_METRIC]

    # Hyperparameter search
    best_trial = trainer.hyperparameter_search(
        # Use value 'minimize' to optimize for validation loss
        direction="minimize",
        n_trials=NUMBER_OF_TRIALS,
        sampler=optuna.samplers.TPESampler(),
        hp_space=hyperparameter_space,
        compute_objective=my_objective,
    )

    # Retrieve best hyperparameters and train
    best_hyperparams = best_trial.hyperparameters
    for n, v in best_hyperparams.items():
        setattr(trainer.args, n, v)

    trainer.train()

    # Save best model
    trainer.save_model(f".{output_path}/best_model")

    # Save best hyperparameters
    with open(f".{output_path}/hyperparameters.json", "w") as file_output:
        json.dump(best_hyperparams, file_output)

    # Evaluate performance on evaluation set
    raw_pred_eval, _, _ = trainer.predict(eval_dataset)

    # Evaluate performance on test_set
    raw_pred_test, _, _ = trainer.predict(test_dataset)

    # Preprocess raw eval predictions
    eval_labels = np.argmax(raw_pred_eval, axis=1)

    # Preprocess raw test predictions
    test_labels = np.argmax(raw_pred_test, axis=1)

    for pred_labels, true_labels, labels_name in zip(
        [test_labels, eval_labels], [y_test, y_val], ["test", "eval"]
    ):

        # Get Precision, Recall, F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=true_labels,
            y_pred=pred_labels,
            beta=1.0,
            average="macro",
            warn_for=tuple(),
            zero_division=0,
        )

        # Get accuracy
        accuracy = accuracy_score(y_true=true_labels, y_pred=pred_labels)

        scores = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

        # Create confusion matrix
        conf_matrix = confusion_matrix(true_labels, pred_labels).ravel()

        confusion_matrix_dict = {
            "true_negative": int(conf_matrix[0]),
            "false_positive": int(conf_matrix[1]),
            "false_negative": int(conf_matrix[2]),
            "true_positive": int(conf_matrix[3]),
        }

        # Save confusion matrix
        with open(
            f".{output_path}/confusion_matrix_{labels_name}.json", "w"
        ) as file_output:
            json.dump(confusion_matrix_dict, file_output)

        # Save scores
        with open(f".{output_path}/scores_{labels_name}.json", "w") as file_output:
            json.dump(scores, file_output)


if __name__ == "__main__":

    # Make sure we are using GPU
    print(f"Utilize GPU: {torch.cuda.is_available()}")

    # Measure training time
    start = time.time()

    # 1. Launch training of the English BERT model
    train_model(model_langs=MODEL_LANGS_ENGLISH, model_type=BERT_EN)

    # 2. Launch training of the multilingual all languages model
    train_model(model_langs=MODEL_LANGS_ALL, model_type=BERT_MULTILANG)

    # 3. Launch training of Italian BERT model
    train_model(model_langs=MODEL_LANGS_IT, model_type=BERT_IT)

    # 4. Launch training of Dutch BERT model
    train_model(model_langs=MODEL_LANGS_NL, model_type=BERT_NL)

    # 5. Launch training of English multilingual BERT and test on Italian and Dutch
    train_model(
        model_langs=MODEL_LANGS_ENGLISH,
        model_type=BERT_MULTILANG,
        excluded_langs_as_test=True,
    )

    # 6. Launch training of Dutch multilingual BERT and test on English and Italian
    train_model(
        model_langs=MODEL_LANGS_NL,
        model_type=BERT_MULTILANG,
        excluded_langs_as_test=True,
    )

    # 7. Launch training of Italian multilingual BERT and test on Dutch and English
    train_model(
        model_langs=MODEL_LANGS_IT,
        model_type=BERT_MULTILANG,
        excluded_langs_as_test=True,
    )

    # 8. Launch training of Italian-English model and test on Dutch
    train_model(
        model_langs=MODEL_LANGS_IT_EN,
        model_type=BERT_MULTILANG,
        excluded_langs_as_test=True,
    )

    # 9. Launch training of Dutch-English model and test on Italian
    train_model(
        model_langs=MODEL_LANGS_NL_EN,
        model_type=BERT_MULTILANG,
        excluded_langs_as_test=True,
    )

    # 10. Launch training Italian-Dutch model and test on English
    train_model(
        model_langs=MODEL_LANGS_IT_NL,
        model_type=BERT_MULTILANG,
        excluded_langs_as_test=True,
    )

    # Print training duration
    print(f"Total training time (s): {time.time() - start}")
