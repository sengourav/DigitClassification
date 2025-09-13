
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


def load_digits_data():
    digits = datasets.load_digits()
    return digits


def prepare_data(digits):
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data


def train_classifier(X_train, y_train):
    clf = svm.SVC(gamma=0.001)
    clf.fit(X_train, y_train)
    return clf


if __name__ == "__main__":
    # --- Already refactored part ---
    digits = load_digits_data()
    data = prepare_data(digits)

    # --- TASK for students ---
    # The code for previewing the first 4 digit images is left here on purpose.
    # Students should refactor it into a function.
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    # Train model
    clf = train_classifier(X_train, y_train)

    # --- TASK for students ---
    # Predict, evaluate, and plot results are still written directly here.
    # Students should refactor them into functions.

    predicted = clf.predict(X_test)

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


def load_digits_data():
    digits = datasets.load_digits()
    return digits


def prepare_data(digits):
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data


def train_classifier(X_train, y_train):
    clf = svm.SVC(gamma=0.001)
    clf.fit(X_train, y_train)
    return clf


if __name__ == "__main__":
    # --- Already refactored part ---
    digits = load_digits_data()
    data = prepare_data(digits)

    # --- TASK for students ---
    # The code for previewing the first 4 digit images is left here on purpose.
    # Students should refactor it into a function.
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    # Train model
    clf = train_classifier(X_train, y_train)

    # --- TASK for students ---
    # Predict, evaluate, and plot results are still written directly here.
    # Students should refactor them into functions.

    predicted = clf.predict(X_test)

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()

