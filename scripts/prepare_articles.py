import pandas as pd
import glob


category = {
1: 'politik',
2: 'wirtschaft',
3: 'finanzen',
4: 'feuilleton',
5: 'sport',
6: 'gesellschaft',
7: 'stil',
8: 'technik-motor',
9: 'wissen',
10: 'reise',
11: 'beruf-chance',
12: 'aktuell'
}


def build_tables(category):
    faz_list = []
    if category is 'aktuell':
        path = f"../data/aktuell/"
    else:
        path = f"../data/{category}/"
    all_files = glob.glob(path + '*.csv')

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        faz_list.append(df)
    faz_articles = pd.concat(faz_list, axis=0, ignore_index=True)
    faz_articles['label'] = category

    return faz_articles


if __name__ == '__main__':
    for key, value in category.items():
        faz = build_tables(value)

