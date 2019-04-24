import pandas as pd
import glob


def train_tables(category):
    raw_train = []

    if category is 'aktuell':
        path = f"../train_data/aktuell/"
    else:
        path = f"../train_data/{category}/"

    all_files = glob.glob(path + '*.csv')

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        raw_train.append(df)
    train_articles = pd.concat(raw_train, axis=0, ignore_index=True)
    train_articles['label'] = category

    return train_articles


def test_tables(category):
    raw_test = []

    if category is 'aktuell':
        path = f"../data/aktuell/"
    else:
        path = f"../data/{category}/"

    all_files = glob.glob(path + '*.csv')

    for filename in all_files:
        data = pd.read_csv(filename, index_col=None, header=0)
        raw_test.append(data)
    test_articles = pd.concat(raw_test, axis=0, ignore_index=True)
    test_articles['label'] = category

    return test_articles


def sample_prediction(pred):
    if pred == 1:
        print('The above text belongs to the category: "Politik"')

    elif pred == 2:
        print('The above text belongs to the category: "Sport"')

    elif pred == 3:
        print('The above text belongs to the category: "Wirtschaft"')

    else:
        print('The above text belongs to one of the other categories.')
