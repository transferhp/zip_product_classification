product_classification
-------


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Installation](#installation)
* [Usage](#usage)
* [Contact](#contact)




<!-- ABOUT THE PROJECT -->
## About The Project
--------


This is my solution to Zip Co Limited Data Science challenge test.

The test aims to implement a model to identify the categories/collection of products based on the features of the products that they have from the crawling data.

More descriptions and requirements about the test can be found [here](./URG-DataScientist-TechnicalChallenge-181121-0045.pdf) 

<!-- GETTING STARTED -->
## Getting Started
-----
### Installation

- Clone the repository by running command:
```sh
git clone https://github.com/transferhp/zip_product_classification.git
```

- Create a virtual environment and go to the root directory of project to run:
```sh
pip install -r requirements.txt
```
to install the required dependencies;

## Solution walkthrough
------
Before running any data analysis or prediction, it is always important to understand the data first. Given our goal is to build a classifier to predict which category a product should belong to, a lot of product relevant information is provided, like product name, text description and pictures etc. 

For simplicity, I focus on using only name and description as inputs for building the classifier. To prepare the text data for the model building, text processing is needed to conduct as it is the very first step for any NLP projects. Some classical preprocessing steps are:

* Remove punctuations, like . , ! $( ) * % @
* Remove number
* Remove stop words
* Lower case
* tokenization
* stemming
* lemmatization

Let's start by reading the data. Without loading everything to explode memory, I choose to read the data from zip file line by line and only process what is needed and save it into a dictionary.

```python
import zipfile
import json

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
``` 

During processing, I have run all listed above steps to clean the text data. More specifically,
```python
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
```

After cleaning all the data, I converted the dictionary into a Pandas DataFrame to assist initial exploratory data analysis. The cleaned data has 125344 rows and 8 columns. 

First step is to check missing values in each column. There is no NaN value in our data as we are handling text data, but it is necessary to check empty string in each column by
```python
cleaned_df.replace('', np.nan).isnull().sum()
```
![cnt_empty_string](./pictures/cnt_empty_string.png)


Due to the nature of craweling data, it is expected to see empty or blank for name or description about a product.

Next we will look what is the distribution for different level of product category.

<center>Product category distribution at level 0</center>

![lvl0](./pictures/lvl0.png)

<center>Product category distribution at level 1</center>

![lvl1](./pictures/lvl1.png)

<center>Product category distribution at level 2 (Generated by internal REGEX classifier)</center>

![lvl2](./pictures/lvl2.png)


From the distribution plot, we can easily find that **fasion** is the largest category in the data, which will also add difficulty to the modelling as the dataset is unbalanced.

Besides category distribution, I also checked the word length of product name and description.
![length_distribution](./pictures/length_distribution.png)

It turns out that for most of products in the crawled data we don't have a long text description though we combined both name and original long description.





More examples about data processing, EDA, clustering visualisation, modelling and referencing can be found in `notebooks` [folder](./notebooks).




<!-- CONTACT -->
## Contact
-----
Author: [Peng Hao](haopengbuaa@gmail.com)

Project Link: [https://github.com/transferhp/zip_product_classification.git](https://github.com/transferhp/zip_product_classification.git)
