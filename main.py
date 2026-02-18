from datasets import load_dataset
import re
from preprocessing_normalisation import preprocess,normalise

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt
import pandas as pd


SEED = 69

# download the AG news dataset
dataset_dict = load_dataset("SetFit/ag_news")
"""
Dataset downloads the test and train split separately
It save them in a dictionary where the first element
is the train dataset, and the second one is the test dataset

they are both Dataset objects containing:
    features: text label label_text
    num_rows: int
"""
# Preprocessing
train_full = dataset_dict["train"].to_pandas()
test = dataset_dict["test"].to_pandas()

train_full["text"] = train_full["text"].str.lower()
test["text"] = test["text"].str.lower()

#applying our preprocessing and normalisation functions from the other file
train_full["text"] = train_full["text"].apply(preprocess)
train_full["text"] = train_full["text"].apply(normalise)
print(train_full.head(20))


# split the training dataset into train and validation
train, dev = train_test_split(
    train_full, test_size=0.16, random_state=SEED, stratify=train_full["label"]
)

print(f"Train size: {len(train)} rows")
print(f"Dev size:   {len(dev)} rows")
print(f"Test size:  {len(test)} rows")

# this is the TF-IDF function
#! the hyperparameters might need to be changed
tf_idf = TfidfVectorizer(
    lowercase=True, stop_words="english", max_features=10000, ngram_range=(1, 2)
)

# apply TF-IDF on the documents
X_train = tf_idf.fit_transform(train["text"])
# do not fit on the dev and an test as it results in leakage
X_dev = tf_idf.transform(dev["text"])
X_test = tf_idf.transform(test["text"])

# print(X_train)

lr = LogisticRegression(random_state=SEED)

svm = LinearSVC(loss="squared_hinge", random_state=SEED)

lr.fit(X_train, train["label"])
svm.fit(X_train, train["label"])

lr_pred = lr.predict(X_test)
svm_pred = svm.predict(X_test)

# Is this not Logistic Regression?
print("Linear Regression metrics:\n")
print(f"Accuracy: {accuracy_score(test['label'], lr_pred):.3f}")
print(f"F1-Score: {f1_score(test['label'], lr_pred, average='macro'):.3f}")
print(f"Confusion Matrix: \n{confusion_matrix(test['label'], lr_pred)}\n")

cm = confusion_matrix(test["label"], lr_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=test["label_text"].unique()
)
disp.plot(xticks_rotation="vertical")
plt.title("Confusion Matrix: Linear Regression")
plt.show()


# And this being Linear Regression?
print("SVM metrics:\n")
print(f"Accuracy: {accuracy_score(test['label'], svm_pred):.3f}")
print(f"F1-Score: {f1_score(test['label'], svm_pred, average='macro'):.3f}")
print(f"Confusion Matrix: \n{confusion_matrix(test['label'], svm_pred)}\n")
cm = confusion_matrix(test["label"], svm_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=test["label_text"].unique()
)
disp.plot(xticks_rotation="vertical")
plt.title("Confusion Matrix: SVM")
plt.show()


# TODO: Collect the misclassified examples and categorize them
# TODO: Get rid of code duplication by putting the metrics in a function

lr_df_predictions = pd.DataFrame(
    {"text": test["text"], "true_label": test["label"], "pred_label": lr_pred}
)

errors = lr_df_predictions[
    lr_df_predictions["true_label"] != lr_df_predictions["pred_label"]
]

print(f"Total Errors: {len(errors)}")
print("Displaying first 20 misclassifications:")

# pd.set_option('display.max_colwidth', 150)
display(errors.head(20))