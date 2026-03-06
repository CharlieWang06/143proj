from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(x, y, test_size: float = 0.3, random_state: int = 42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def scale_features(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler


def train_logistic_regression(x_train, y_train) -> LogisticRegression:
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, x_test, y_test) -> tuple[str, list[list[int]]]:
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    return report, cm
