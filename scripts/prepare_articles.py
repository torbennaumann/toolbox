import pandas as pd
import glob


category = {
    1: 'politik', 2: 'wirtschaft', 3: 'finanzen', 4: 'feuilleton', 5: 'sport', 6: 'gesellschaft', 7: 'stil',
    8: 'technik-motor', 9: 'wissen', 10: 'reise', 11: 'beruf-chance', 12: 'aktuell'
}


def build_tables(category):
    raw_articles = []
    if category is 'aktuell':
        path = f"../new_data/aktuell/"
    else:
        path = f"../new_data/{category}/"

    all_files = glob.glob(path + '*.csv')

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        raw_articles.append(df)
    faz_articles = pd.concat(raw_articles, axis=0, ignore_index=True)
    faz_articles['label'] = category

    return faz_articles


if __name__ == '__main__':
    frames = []
    for key, value in category.items():
        raw_faz = build_tables(value)
        frames.append(raw_faz)
        faz = pd.concat(frames, axis=0, ignore_index=True)
