from datasets import load_dataset
from preprocessing_normalisation import preprocess, normalise

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    f1_score,
    confusion_matrix,
    classification_report,
    hinge_loss,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt
import pandas as pd


SEED = 69
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def calculate_metrics(real: pd.DataFrame, pred: list,  model: str) -> None:
    print(f"\n{model} metrics:\n")
    print(f"Accuracy: {accuracy_score(real['label'], pred):.3f}")
    print(f"F1-Score: {f1_score(real['label'], pred, average='macro'):.3f}")
    print(f"Confusion Matrix: \n{confusion_matrix(real['label'], pred)}\n")

    cm = confusion_matrix(real['label'], pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=real['label_text'].unique()
    )
    disp.plot(xticks_rotation="vertical")
    plt.title(f"Confusion Matrix: {model}")
    plt.show()

    predictions = pd.DataFrame({
        "text": real["text"],
        "true_label": real["label"].map(LABELS),
        "pred_label": pd.Series(pred).map(LABELS),
    })

    errors = predictions[
        predictions["true_label"] != predictions["pred_label"]
    ]

    print(f"\nTotal Errors: {len(errors)}")
    print("Displaying first 20 misclassifications:\n")

    for i, doc in errors.head(20).iterrows():
        print(f"Article number {i}:")
        print(f"TRUE: {doc['true_label']} | PRED: {doc['pred_label']}")
        print(f"TEXT: {doc['text']}")
        print("-" * 80)


def hyperparameter_tuning(
        train: pd.DataFrame,
        dev: pd.DataFrame,
        model: LogisticRegression | LinearSVC,
        epochs: int = 15,
        ) -> TfidfVectorizer:

    max_features = 1000
    picked_tf_idf = None
    best = float('inf')

    for i in range(epochs):
        print(f"\nEpoch {i}:")

        tf_idf = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, 2)
        )

        # apply TF-IDF on the documents
        X_train = tf_idf.fit_transform(train["text"])
        X_dev = tf_idf.transform(dev["text"])

        # train the model
        model.fit(X_train, train["label"])

        if isinstance(model, LogisticRegression):
            train_pred = model.predict_proba(X_train)
            dev_pred = model.predict_proba(X_dev)
            # calculate the log loss
            test_loss = log_loss(train["label"], train_pred)
            dev_loss = log_loss(dev["label"], dev_pred)

        else:
            train_pred = model.decision_function(X_train)
            dev_pred = model.decision_function(X_dev)
            # calculate the squared hinge loss
            test_loss = hinge_loss(train["label"], train_pred)**2
            dev_loss = hinge_loss(dev["label"], dev_pred)**2

        print(f"Test Loss: {test_loss} | Dev Loss: {dev_loss}")
        # save the TF-IDF that leads to the smallest loss
        if dev_loss < best:
            best, picked_tf_idf = dev_loss, tf_idf

        max_features += 2000

    return picked_tf_idf


def main() -> None:

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
    train_full["text"] = train_full["text"].apply(preprocess).apply(normalise)
    test["text"] = test["text"].apply(preprocess).apply(normalise)

    print(train_full.head(5))

    # split the training dataset into train and validation
    train, dev = train_test_split(
        train_full,
        test_size=0.16,
        random_state=SEED,
        stratify=train_full["label"]
    )

    print(f"\nTrain size: {len(train)} rows")
    print(f"Dev size:   {len(dev)} rows")
    print(f"Test size:  {len(test)} rows")

    lr = LogisticRegression(random_state=SEED, solver="saga")
    #svm = LinearSVC(loss="squared_hinge", random_state=SEED)

    tf_idf = hyperparameter_tuning(
        train=train,
        dev=dev,
        model=lr
    )

    # apply TF-IDF on the documents
    X_train = tf_idf.fit_transform(train["text"])
    # do not fit on the dev and on test as it results in leakage
    X_test = tf_idf.transform(test["text"])

    lr.fit(X_train, train["label"])
    lr_pred = lr.predict(X_test)
    calculate_metrics(test, lr_pred, "Logistic Regression")


    #svm.fit(X_train, train["label"])
    #svm_pred = svm.predict(X_test)
    #calculate_metrics(test, svm_pred, "SVM")


if __name__ == "__main__":
    main()
