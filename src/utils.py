from sklearn.metrics import classification_report


def evaluate_on_data(data_train, data_test):
    X_train, y_train = data_train
    X_test, y_test = data_test

    def evaluate_classifier(clf, fit_clf=True, **clf_params):
        if fit_clf:
            clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = clf.score(X_test, y_test)
        report = classification_report(y_test, y_pred)
        return acc, report

    return evaluate_classifier
