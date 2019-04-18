import matplotlib.pyplot as plt
import pandas as pd
import re
import string
from wordcloud import WordCloud


def get_words():
    rgx_singles = re.compile("([\w][\w']*[\w])")
    rgx_doubles = re.compile("([\w][\w']*[\w] +[\w][\w']*[\w])")
    rgx_triples = re.compile("([\w][\w']*[\w] +[\w][\w']*[\w] +[\w][\w']*[\w])")
    translator = str.maketrans('', '', string.punctuation)

    field = []
    for index, row in pd.DataFrame.iterrows():
        line = row['detailed'].translate(translator).lower()
        field += rgx_singles.findall(line)
        field += rgx_doubles.findall(line)
        field += rgx_triples.findall(line)

    word_series = pd.Series({field}).str.cat(sep=' ')
    return word_series


def get_cloud(word_series):
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200, stopwords=, background_color='white',
                          colormap='twilight').generate(word_series)
    plt.figure(figsize=(20, 10), )
    plt.imshow(wordcloud, interpolation='bilinear', )
    plt.axis("off")
    plt.show()
    return wordcloud