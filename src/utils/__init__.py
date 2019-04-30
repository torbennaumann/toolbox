import pandas as pd
from wordcloud import WordCloud
import glob
import matplotlib.pyplot as plt


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


def print_classification(prediction):
    if prediction == 1:
        print('The text above belongs to the category: "Politik"')
    elif prediction == 2:
        print('The text above belongs to the category: "Sport"')
    elif prediction == 3:
        print('The text above belongs to the category: "Wirtschaft"')
    elif prediction == 4:
        print('The text above belongs to the category: "Feuilleton"')
    elif prediction == 5:
        print('The text above belongs to the category: "Finanzen"')
    elif prediction == 6:
        print('The text above belongs to the category: "Gesellschaft"')
    else:
        print('The text above belongs to one of these categories: '
              '"Beruf-Chance", "Reise", "Stil", "Technik-Motor" or "Wissen".')


def get_wordcloud(df, stopwordlist):
    wordcloud = WordCloud(width=1600, height=800, max_font_size=150, max_words=100, stopwords=stopwordlist,
                          background_color='white',
                          colormap='twilight').generate(df)
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
