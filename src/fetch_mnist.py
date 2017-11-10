from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocessed_mnist(random_state, preprocess='standardize'):
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
    elif preprocess == 'scale_to_unit_cube':
        return scale_to_unit_cube(X, y, random_state)
    else:
        raise NotImplementedError('Unsupported preprocessing method: {}'.format(preprocess))


def scale_to_unit_cube(X_raw, y, random_state):
    X_mean_subtracted = X_raw - X_raw.mean()
    scaler = MinMaxScaler(feature_range=(-1, 1))

    X = scaler.fit_transform(X_mean_subtracted)
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
