"""
Script preprocesses downloaded and scraped datasets and combines them into ready-to-use training datasets

"""

# Import libraries
import pandas as pd
import os
import numpy as np

# Get name of current working direct
cwd = os.getcwd()


# Dutch Data
def dutch_data() -> pd.DataFrame:

    # De Speld News Data - Scraped from https://speld.nl/sitemap_index.xml
    de_speld_files = []
    de_speld_files_path = f"{cwd}/de_speld/"

    for (_, _, filenames) in os.walk(de_speld_files_path):
        for filename in filenames:
            if filename.endswith(".csv"):
                de_speld_files.append(filename)

    df_despeld = pd.DataFrame()

    for item in de_speld_files:
        df_temp = pd.read_csv(de_speld_files_path + item)
        df_despeld = pd.concat([df_despeld, df_temp])

    df_despeld = df_despeld.drop_duplicates(subset=["article_url"])
    df_despeld["is_sarcastic"] = [1] * len(df_despeld)
    df_despeld = df_despeld.filter(
        items=["article_url", "article_title", "is_sarcastic", "title_length"]
    )
    df_despeld["title_length"] = df_despeld.article_title.apply(lambda x: len(x))
    df_despeld = df_despeld.loc[df_despeld["title_length"] > 45]
    df_despeld = df_despeld.loc[df_despeld["title_length"] < 185]
    median_sarcastic_title_length_nl = np.median(df_despeld["title_length"])
    print(
        f"Median length of sarcastic title in Dutch: {median_sarcastic_title_length_nl}"
    )

    # NOS News Data - Downloaded from Kaggle: https://www.kaggle.com/datasets/maxscheijen/dutch-news-articles
    nos_files_path = f"{cwd}/datasets/nos/"
    nos_file_name = None

    for (_, _, filenames) in os.walk(nos_files_path):
        for filename in filenames:
            if filename.endswith(".csv"):
                nos_file_name = filename

    df_nos = pd.read_csv(nos_files_path + nos_file_name)
    df_nos = df_nos.drop_duplicates(subset=["url"])
    df_nos["title_length"] = df_nos.title.apply(lambda x: len(x))
    df_nos = df_nos.loc[df_nos["title_length"] > 45]
    df_nos = df_nos.loc[df_nos["title_length"] < 185]
    df_nos = df_nos.sample(n=15000)
    median_non_sarcastic_title_length_nl = np.median(df_nos["title_length"])
    print(
        f"Median length of non-sarcastic title in Dutch: {median_non_sarcastic_title_length_nl}"
    )

    # df_nos = df_nos.sample(n = len(df_despeld))
    df_nos = df_nos.rename(columns={"title": "article_title", "url": "article_url"})
    df_nos["is_sarcastic"] = [0] * len(df_nos)
    df_nos = df_nos.filter(
        items=["article_url", "article_title", "is_sarcastic", "title_length"]
    )

    # Combine Dutch news set
    df_nl_data = pd.concat([df_nos, df_despeld])
    df_nl_data["lang"] = ["nl"] * len(df_nl_data)
    df_nl_data["article_title"] = df_nl_data.article_title.apply(lambda x: x.lower())
    df_nl_data.to_csv("./training_data/dutch_data_set.csv", index=False)

    return df_nl_data


# Italian Data
def italian_data() -> pd.DataFrame:

    # Lercio News Data - Scraped from https://www.lercio.it/sitemap_index.xml
    lercio_files = []
    lercio_files_path = f"{cwd}/lercio/"

    for (_, _, filenames) in os.walk(lercio_files_path):
        for filename in filenames:
            if filename.endswith(".csv"):
                lercio_files.append(filename)

    df_lercio = pd.DataFrame()

    for item in lercio_files:
        df_temp = pd.read_csv(lercio_files_path + item)
        df_lercio = pd.concat([df_lercio, df_temp])

    df_lercio = df_lercio.drop_duplicates(subset=["article_url"])
    df_lercio["article_title"] = df_lercio.article_title.str.replace("| Lercio", "")
    df_lercio["article_title"] = df_lercio.article_title.str.replace("|", "")
    df_lercio["title_length"] = df_lercio.article_title.apply(lambda x: len(x))
    df_lercio = df_lercio.loc[df_lercio["title_length"] > 45]
    df_lercio = df_lercio.loc[df_lercio["title_length"] < 185]
    df_lercio = df_lercio[
        df_lercio["article_url"].str.endswith(tuple([".jpeg", ".png"])) == False
    ]
    df_lercio["is_sarcastic"] = [1] * len(df_lercio)

    median_sarcastic_title_length_it = np.median(df_lercio["title_length"])
    print(
        f"Median length of sarcastic title (Lercio) in Italian: {median_sarcastic_title_length_it}"
    )
    df_lercio = df_lercio.filter(
        items=["article_url", "article_title", "is_sarcastic", "title_length"]
    )

    # Il Giornale News Data - Scraped from https://www.ilgiornale.it/sitemap/indice.xml
    giornale_files = []
    giornale_files_path = f"{cwd}/giornale/"

    for (_, _, filenames) in os.walk(giornale_files_path):
        for filename in filenames:
            if filename.endswith(".csv"):
                giornale_files.append(filename)

    df_giornale = pd.DataFrame()

    for item in giornale_files:
        df_temp = pd.read_csv(giornale_files_path + item)
        df_giornale = pd.concat([df_giornale, df_temp])

    df_giornale = df_giornale.drop_duplicates(subset=["article_url"])
    df_giornale["title_length"] = df_giornale.article_title.apply(lambda x: len(str(x)))
    df_giornale = df_giornale.loc[df_giornale["title_length"] > 45]
    df_giornale = df_giornale.loc[df_giornale["title_length"] < 185]
    df_giornale = df_giornale.sample(n=15000)
    df_giornale = df_giornale[
        df_giornale["article_url"].str.endswith(tuple([".jpeg", ".png"])) == False
    ]
    df_giornale["is_sarcastic"] = [0] * len(df_giornale)

    median_sarcastic_title_length_it = np.median(df_giornale["title_length"])
    print(
        f"Median length of sarcastic title (Il Giornale) in Italian: {median_sarcastic_title_length_it}"
    )
    df_giornale = df_giornale.filter(
        items=["article_url", "article_title", "is_sarcastic", "title_length"]
    )

    # Combine Italian news set
    df_it_data = pd.concat([df_lercio, df_giornale])
    df_it_data["lang"] = ["it"] * len(df_it_data)
    df_it_data["article_title"] = df_it_data.article_title.apply(lambda x: x.lower())
    df_it_data.to_csv("./training_data/italian_data_set.csv", index=False)

    return df_it_data


# English Data
def english_data() -> pd.DataFrame:

    # TheOnion and Huffington Post - The Sarcasm Detection Dataset: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
    en_data_path = f"{cwd}/datasets/english_data/"
    en_data_filenames = []

    for (_, _, filenames) in os.walk(en_data_path):
        for filename in filenames:
            if filename.endswith(".json"):
                en_data_filenames.append(filename)

    df_en_data = pd.DataFrame()

    for item in en_data_filenames:

        df_temp = pd.read_json(en_data_path + item, lines=True)
        df_en_data = pd.concat([df_en_data, df_temp])

    df_en_data = df_en_data.drop_duplicates(subset=["article_link"])
    df_en_data = df_en_data.rename(
        columns={"headline": "article_title", "article_link": "article_url"}
    )
    df_en_data["title_length"] = df_en_data.article_title.apply(lambda x: len(x))

    df_en_data["lang"] = ["en"] * len(df_en_data)
    df_en_data = df_en_data.loc[df_en_data["title_length"] > 45]
    df_en_data = df_en_data.loc[df_en_data["title_length"] < 185]

    df_en_sarcastic = df_en_data.loc[df_en_data["is_sarcastic"] == 1]
    df_en_non_sarcastic = df_en_data.loc[df_en_data["is_sarcastic"] == 0]

    median_sarcastic_title_length_en = np.median(df_en_sarcastic["title_length"])
    print(
        f"Median length of sarcastic title in English: {median_sarcastic_title_length_en}"
    )

    median_non_sarcastic_title_length_en = np.median(
        df_en_non_sarcastic["title_length"]
    )
    print(
        f"Median length of non-sarcastic title in English: {median_non_sarcastic_title_length_en}"
    )

    df_en_data = df_en_data.filter(
        items=["article_url", "article_title", "is_sarcastic", "lang", "title_length"]
    )
    df_en_data.to_csv("./training_data/english_data_set.csv", index=False)

    return df_en_data


# Combine datasets together to make multilingual dataset
def multilingual_dataset(
    en_data: pd.DataFrame, it_data: pd.DataFrame, nl_data: pd.DataFrame
) -> pd.DataFrame:

    df_multilingual = pd.concat([en_data, it_data, nl_data])
    df_multilingual.to_csv("./training_data/multilang_sarcasm_dataset.csv", index=False)

    return df_multilingual


if __name__ == "__main__":

    # Create Dutch training set
    df_nl_data = dutch_data()

    # Create Italian training set
    df_it_data = italian_data()

    # Create English training set
    df_en_data = english_data()

    # Create multilingual dataset
    df_multilingual = multilingual_dataset(
        en_data=df_en_data, it_data=df_it_data, nl_data=df_nl_data
    )
