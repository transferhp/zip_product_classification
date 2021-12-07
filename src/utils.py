import zipfile
import json
import string
import re

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")


def stemming_words(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]


def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


def text_processing(input_str):
    # Lowercase
    input_str = input_str.lower()
    # Remove numbers
    clean_str = re.sub(r"\d+", "", input_str)
    # Remove special symbols
    clean_str = re.sub(r"\W+", " ", clean_str)
    # Remove punctuation
    clean_str = clean_str.translate(str.maketrans("", "", string.punctuation))
    # Tokenize words
    list_words = word_tokenize(clean_str)
    # Remove stop words
    list_words = [word for word in list_words if word not in stopwords.words("english")]
    # Stem
    list_words = stemming_words(list_words)
    # Lem
    list_words = lemmatize_words(list_words)
    return " ".join(list_words)


def load_raw_data(file_path):
    data = {"id": [], "name": [], "desc": [], "cat0": [], "cat1": [], "cat2": []}

    with zipfile.ZipFile(f"{file_path}") as z:
        with z.open("exercise3.jl") as f:
            for line in f:
                product_info = json.loads(line.strip())
                data["id"].append(product_info["_id"])
                if "product_name" in product_info["_source"]:
                    data["name"].append(
                        text_processing(product_info["_source"]["product_name"])
                    )
                else:
                    data["name"].append("")
                if "long_description" in product_info["_source"]:
                    data["desc"].append(
                        text_processing(product_info["_source"]["long_description"])
                    )
                else:
                    data["desc"].append("")

                cat0 = ", ".join(product_info["_source"]["e_cat_l0"]).lower()
                cat1 = ", ".join(product_info["_source"]["e_cat_l1"]).lower()
                cat2 = ", ".join(product_info["_source"]["e_cat_l2"]).lower()
                data["cat0"].append(cat0)
                data["cat1"].append(cat1)
                data["cat2"].append(cat2)

        print("Logging Info - Data Size:", len(data["id"]))
        assert (
            len(data["id"])
            == len(data["name"])
            == len(data["desc"])
            == len(data["cat0"])
            == len(data["cat1"])
            == len(data["cat2"])
        )
        return data
