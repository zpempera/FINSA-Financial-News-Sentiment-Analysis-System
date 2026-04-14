# FINSA: Financial Sentiment Analysis with Classical and Embedding-Based Text Representations

## Overview

This project focuses on multiclass financial sentiment classification. The goal is to classify financial statements into sentiment categories using a combination of text preprocessing, exploratory data analysis, feature engineering, multiple text representations, and several machine learning models.

The workflow compares traditional sparse representations such as TF-IDF with dense embeddings such as Word2Vec and BERT, and also evaluates whether adding handcrafted linguistic features improves performance.

## Open in Deep Note
[Open in Deep Note](https://deepnote.com/workspace/Zofias-workspace-a73f9631-ee62-4aa8-bf24-7af5f61bd20b/project/FINSA-f03c288d-2501-42f4-8a6c-a5e5bfb6c716/notebook/Notebook-1-f72feff0a9e04f1c88f9a14903e18134?utm\_source=share-modal\&utm\_medium=product-shared-content\&utm\_campaign=notebook\&utm\_content=f03c288d-2501-42f4-8a6c-a5e5bfb6c716)
## Project Goals

The main objectives of this project are:

- to preprocess and normalize financial text data
- to explore linguistic and statistical differences between sentiment classes
- to compare multiple text representations for sentiment classification
- to benchmark several machine learning models on the same dataset
- to identify the best-performing approach and interpret its behaviour

## Dataset

The dataset consists of 5,842 labeled financial text samples. Each observation contains:

- a financial sentence
- a sentiment label

The dataset contains no missing values and is imbalanced, with the neutral class being the largest.

## Text Preprocessing

A custom preprocessing pipeline was implemented to prepare the financial text for modeling. The steps include:

- lowercasing
- URL and email removal
- currency normalization
- percentage normalization
- tokenization
- POS-aware lemmatization
- stopword filtering with preservation of selected financial terms

The pipeline also extracts handcrafted linguistic features such as:

- word count
- character count
- average word length
- sentence count
- exclamation and question counts
- uppercase ratio
- financial term count
- positive word count
- negative word count

## Exploratory Data Analysis

The project includes visual and statistical analysis of the dataset, including:

- sentiment label distribution
- text length distribution by sentiment
- word count comparison across classes
- average financial term usage by sentiment
- positive vs negative keyword analysis
- word clouds for each class

## Text Representations

The following text representations were tested:

### 1. TF-IDF
- unigram
- bigram
- trigram

### 2. Word2Vec
A skip-gram Word2Vec model trained on the dataset, with sentence embeddings computed as the average of word vectors.

### 3. BERT Embeddings
BERT embeddings extracted from the CLS token using `bert-base-uncased` (computed on a subset due to computational cost).

### 4. TF-IDF + Linguistic Features
Combination of TF-IDF vectors with handcrafted linguistic features.

## Models Evaluated

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Additionally, a soft-voting ensemble was tested.

## Evaluation

Models were evaluated using:

- Accuracy
- F1-Macro
- F1-Weighted
- classification report
- confusion matrix
- feature importance analysis
- learning curves
- cross-validation

## Results

Best performing model:

- **Model:** Logistic Regression  
- **Representation:** TF-IDF + Linguistic Features  
- **Accuracy:** 0.696  
- **F1-Macro:** 0.653  

Top 3 models:

1. Logistic Regression (TF-IDF + Linguistic Features)
2. Logistic Regression (TF-IDF trigram)
3. Logistic Regression (TF-IDF)

Key observations:

- Classical TF-IDF representations outperformed Word2Vec and BERT in this setup
- Adding linguistic features improved performance
- Logistic Regression consistently performed best
- The negative class was the hardest to classify

