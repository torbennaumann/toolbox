import feedparser
import pandas as pd
import datetime


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


def scrape_article(category):
    if category is not 'aktuell':
        entries = feedparser.parse(f"https://www.faz.net/rss/aktuell/{category}/")
    else:
        entries = feedparser.parse(f"https://www.faz.net/rss/aktuell/")

    faz = pd.DataFrame(entries.entries)
    faz.drop(['guidislink', 'id', 'links', 'published_parsed', 'summary_detail', 'title_detail'], axis=1, inplace=True)
    return faz


def get_subtext(faz):
    descriptions = []

    for index, text in enumerate(faz.summary):
        text = faz.summary[index].split('title="')[1].split('" alt=')[0] + faz.summary[index].split('<p>')[1].split('</p>')[0]
        descriptions.append(text)

    subtext = pd.DataFrame(descriptions, columns=['detailed'])
    faz_agg = pd.concat([faz, subtext], axis=1)
    faz_agg.drop(['summary'], axis=1, inplace=True)

    return faz_agg


def save_data(faz_agg, category):
    name = "table_" + str(datetime.datetime.now()) + ".csv"

    if category is 'aktuell':
        path = f"../data/aktuell/" + name
    else:
        path = f"../data/{category}/" + name

    faz_agg.to_csv(path, index=False)


if __name__ == '__main__':
    for key, value in category.items():
        try:
            faz = scrape_article(value)
            faz_agg = get_subtext(faz)
            save_data(faz_agg, value)
            print(f'Saving most recent articles for you from "{value}".')
        except:
            print(f'ErrorFetching: {value}')
