import pandas as pd


def create_stopwords_plain():
    stopwords = []

    file = open('/Users/torben/PycharmProjects/toolbox/stopwords/german_stopwords_plain.txt', 'r')
    for line in file:
        line = line
        stopwords.append(line.replace('\n', ''))

    plain = pd.DataFrame(stopwords)

    return plain


def create_stopwords_full():
    stopwords_full = []

    file = open('/Users/torben/PycharmProjects/toolbox/stopwords/german_stopwords_full.txt', 'r')
    for line in file:
        stopwords_full.append(line.replace('\n', ''))

    full = pd.DataFrame(stopwords_full)

    return full


def create_list(plain, full):
    stopword = pd.concat([full, plain])
    stopword = stopword.drop_duplicates()
    stopword.columns = ['words']
    return stopword


def save_data(stopword):
    name = "stopwords.csv"
    path = f'/Users/torben/PycharmProjects/toolbox/stopwords/' + name
    stopword.to_csv(path, index=False)


if __name__ == '__main__':
    swlp = create_stopwords_plain()
    swlf = create_stopwords_full()
    total = create_list(swlp,swlf)
    save_data(total)
