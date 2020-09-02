import pandas as pd
import idx2numpy

from sklearn.model_selection import train_test_split


def read_wine_data(test_size=.2, random_state=None, task="class"):
    assert task in ("class", "reg")
    if task == "class":
        target_col = "type_red"
    else:
        target_col = "quality"

    red = pd.read_csv("data/wine-quality/winequality-red.csv", sep=";")
    red["type"] = "red"
    white = pd.read_csv("data/wine-quality/winequality-white.csv", sep=";")
    white["type"] = "white"

    df = pd.concat([red, white])
    df.loc[:, "type"] = (df["type"] == "red") * 1
    df = df.rename({"type": "type_red"}, axis=1)

    x_train, x_test = train_test_split(df, test_size=test_size, stratify=df[target_col], random_state=random_state)
    y_train = x_train[target_col]
    y_test = x_test[target_col]
    x_train = x_train.drop(target_col, axis=1)
    x_test = x_test.drop(target_col, axis=1)

    return {"X": x_train, "y": y_train, "X_test": x_test, "y_test": y_test}


def read_mnist_data():
    x_train = idx2numpy.convert_from_file("data/mnist/train-images.idx3-ubyte").reshape(60000, 28 * 28)
    y_train = idx2numpy.convert_from_file("data/mnist/train-labels.idx1-ubyte")
    x_test = idx2numpy.convert_from_file("data/mnist/t10k-images.idx3-ubyte").reshape(10000, 28 * 28)
    y_test = idx2numpy.convert_from_file("data/mnist/t10k-labels.idx1-ubyte")

    return {"X": x_train, "y": y_train, "X_test": x_test, "y_test": y_test}


def read_bike_data(test_size=.2, random_state=None):
    target_col = "cnt"
    
    df_bike = pd.read_csv("data/Bike-Sharing-Dataset/hour.csv", index_col=0)
    df_bike = df_bike.drop(["dteday", "casual", "registered"], axis=1)
    
    x_train, x_test = train_test_split(df_bike, test_size=test_size, random_state=random_state)
    y_train = x_train[target_col]
    y_test = x_test[target_col]
    x_train = x_train.drop(target_col, axis=1)
    x_test = x_test.drop(target_col, axis=1)

    return {"X": x_train, "y": y_train, "X_test": x_test, "y_test": y_test}
