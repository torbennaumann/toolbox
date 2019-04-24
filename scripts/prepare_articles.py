import pandas as pd
import glob


category = {
    1: 'politik', 2: 'wirtschaft', 3: 'finanzen', 4: 'feuilleton', 5: 'sport', 6: 'gesellschaft', 7: 'stil',
    8: 'technik-motor', 9: 'wissen', 10: 'reise', 11: 'beruf-chance'
            }


def train_tables(category):
    raw_articles = []

    path = f"../data/{category}/"

    all_files = glob.glob(path + '*.csv')

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        raw_articles.append(df)
    train_articles = pd.concat(raw_articles, axis=0, ignore_index=True)
    train_articles['label'] = category

    return train_articles


def test_tables(category):
    raw = []

    path = f"../train_data/{category}/"

    all_files = glob.glob(path + '*.csv')

    for filename in all_files:
        data = pd.read_csv(filename, index_col=None, header=0)
        raw.append(data)
    test_articles = pd.concat(raw, axis=0, ignore_index=True)
    test_articles['label'] = category

    return test_articles


if __name__ == '__main__':
    train = []
    test = []
    for key, value in category.items():
        raw_train = train_tables(value)
        raw_test = test_tables(value)
        train.append(raw_train)
        test.append(raw_test)
        training_data = pd.concat(train, axis=0, ignore_index=True)
        testing_data = pd.concat(test, axis=0, ignore_index=True)
    print('Creating test and train data frames for you!')
