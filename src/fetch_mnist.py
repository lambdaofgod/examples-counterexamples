import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler


def preprocessed_mnist(random_state):
    """
    Fetches mnist dataset and then applies standard scaling

    WARNING: This used builtin fetch_mldata function which
    connects to the internet and downloads data for the first
    time it is called
    """
    mnist = fetch_mldata('MNIST original')

    X = mnist['data']
    y = mnist['target'].astype(int)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state) 

    sscaler = StandardScaler()
    X_train = sscaler.fit_transform(X_train_raw)
    X_test = sscaler.transform(X_test_raw)

    return (X_train, X_test, y_train, y_test)
