from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocessed_mnist(random_state, preprocess='standardize', scale_min=0, scale_max=1):
    """
    Fetches mnist dataset and then applies standard scaling

    WARNING: This used builtin fetch_mldata function which
    connects to the internet and downloads data for the first
    time it is called
    """
    mnist = fetch_mldata('MNIST original')

    X = mnist['data']
    y = mnist['target'].astype(int)
    if preprocess == 'standardize':
        return standardize(X, y, random_state)
    elif preprocess == 'min_max_scale':
        return min_max_scale(X, y, random_state, scale_min, scale_max)
    else:
        raise NotImplementedError('Unsupported preprocessing method: {}'.format(preprocess))


def min_max_scale(X_raw, y, random_state, scale_min, scale_max):
    X_mean_subtracted = X_raw - X_raw.mean()
    scaler = MinMaxScaler(feature_range=(scale_min, scale_max))

    X = scaler.fit_transform(X_raw)
    return (train_test_split(
        X,
        y,
        test_size=0.2, stratify=y,
        random_state=random_state))


def standardize(X_raw, y, random_state):
    scaler = StandardScaler()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2, stratify=y,
        random_state=random_state)
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test
