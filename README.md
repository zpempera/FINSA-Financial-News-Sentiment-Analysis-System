\# FINSA: Financial Sentiment Analysis with Classical and Embedding-Based Text Representations



\## Overview



This project focuses on multiclass financial sentiment classification. The goal is to classify financial statements into sentiment categories using a combination of text preprocessing, exploratory data analysis, feature engineering, multiple text representations, and several machine learning models.



The workflow compares traditional sparse representations such as TF-IDF with dense embeddings such as Word2Vec and BERT, and also evaluates whether adding handcrafted linguistic features improves performance.



\## See in DeepNote



\[See in DeepNote](https://deepnote.com/workspace/Zofias-workspace-a73f9631-ee62-4aa8-bf24-7af5f61bd20b/project/FINSA-f03c288d-2501-42f4-8a6c-a5e5bfb6c716/notebook/Notebook-1-f72feff0a9e04f1c88f9a14903e18134?utm\_source=share-modal\&utm\_medium=product-shared-content\&utm\_campaign=notebook\&utm\_content=f03c288d-2501-42f4-8a6c-a5e5bfb6c716) 



\## Project Goals



The main objectives of this project are:



\- to preprocess and normalize financial text data

\- to explore linguistic and statistical differences between sentiment classes

\- to compare multiple text representations for sentiment classification

\- to benchmark several machine learning models on the same dataset

\- to identify the best-performing approach and interpret its behaviour



\## Dataset



The dataset consists of 5,842 labeled financial text samples. Each observation contains:



\- a financial sentence

\- a sentiment label



The data is loaded from a local CSV file and renamed into the following schema:



\- `sentence`

\- `label`



No missing values were found in either column. The class distribution is imbalanced, with the neutral class being the largest. The summary shown in the notebook reports class sizes of 860, 3130, and 1852 samples across the three sentiment labels.



\## Text Preprocessing



A custom preprocessing pipeline was implemented to prepare the financial text for modeling. The steps include:



\- lowercasing

\- URL and email removal

\- currency normalization

\- percentage normalization

\- tokenization

\- POS-aware lemmatization

\- stopword filtering with preservation of selected financial terms



The pipeline also extracts handcrafted linguistic features such as:



\- word count

\- character count

\- average word length

\- sentence count

\- exclamation and question counts

\- uppercase ratio

\- financial term count

\- positive word count

\- negative word count



This makes the project stronger than a simple notebook using only raw TF-IDF features.



\## Exploratory Data Analysis



The project includes visual and statistical analysis of the dataset, including:



\- sentiment label distribution

\- text length distribution by sentiment

\- word count comparison across classes

\- average financial term usage by sentiment

\- average positive and negative keyword counts

\- word clouds for each class



The analysis shows meaningful class-level differences in vocabulary, length, and keyword usage. For example, negative examples contain more negative cue words on average, while positive examples contain more positive cue words. The visualizations and summary statistics for these patterns are shown in the notebook.



\## Text Representations



The following text representations were tested:



\### 1. TF-IDF

Several TF-IDF variants were evaluated:

\- unigram

\- bigram

\- trigram



\### 2. Word2Vec

A skip-gram Word2Vec model was trained on the training portion of the processed corpus, and sentence embeddings were created by averaging word vectors.



\### 3. BERT Embeddings

BERT embeddings were extracted from the CLS token using `bert-base-uncased`. Due to computational constraints, this representation was tested on a reduced subset of the data.



\### 4. TF-IDF + Linguistic Features

A combined feature space was built by concatenating TF-IDF vectors with scaled handcrafted linguistic features.



\## Models Evaluated



For each representation, the following classifiers were trained and evaluated:



\- Logistic Regression

\- Random Forest

\- Support Vector Machine



An additional soft-voting ensemble was also tested on the TF-IDF representation.



\## Evaluation



The models were evaluated using:



\- Accuracy

\- F1-Macro

\- F1-Weighted

\- classification report

\- confusion matrix

\- feature importance analysis

\- learning curve

\- cross-validation



This setup allows both predictive comparison and interpretability.



\## Main Results



The best overall model was:



\- \*\*Model:\*\* Logistic Regression

\- \*\*Representation:\*\* TF-IDF\_Combined

\- \*\*Accuracy:\*\* 0.696

\- \*\*F1-Macro:\*\* 0.653

\- \*\*F1-Weighted:\*\* 0.706



Top 3 performing models:



1\. Logistic Regression (TF-IDF\_Combined) — F1-Macro: 0.653, Accuracy: 0.696

2\. Logistic Regression (TF-IDF\_trigram) — F1-Macro: 0.634, Accuracy: 0.686

3\. Logistic Regression (TF-IDF) — F1-Macro: 0.631, Accuracy: 0.683



The ensemble model performed worse than the best single model, reaching F1-Macro 0.579 and Accuracy 0.674.



\## Interpretation



The results suggest that:



\- classical sparse representations were more effective than Word2Vec and BERT in this setup

\- adding handcrafted linguistic features to TF-IDF improved performance

\- Logistic Regression was consistently stronger than Random Forest and SVM across most representations

\- the negative class remained the hardest to classify, as shown by the best model’s classification report and confusion matrix 



The feature importance analysis for the best model highlighted terms such as `currency`, `rise`, `decrease`, `increase`, `fell`, `buy`, and `grow` as especially informative for prediction.

