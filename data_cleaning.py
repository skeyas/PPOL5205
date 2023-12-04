import os
from collections import Counter
from functools import reduce
from heapq import nlargest

import wordcloud
from bokeh.io import show
from holoviews import Chord, opts, dim
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords, sentiwordnet as swn
from nltk import bigrams, trigrams, word_tokenize
import nltk
import spacy
import re

import pandas as pd

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import PCA

from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpld3

from sklearn.feature_extraction.text import CountVectorizer

import holoviews as hv


class Processor:
    @staticmethod
    def pre_process(string):
        manual_exclusions = ["people", "one", "country", "day", "war", "u", "s", "would",
                             "could", "also", "n't"]
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(string)
        stop_words = set(stopwords.words('english'))
        return [lemmatizer.lemmatize(t.lower()) for t in tokens if
                t.lower() not in stop_words and t.lower() not in punctuation and
                t.lower() not in manual_exclusions and len(t.lower()) > 1]

    @staticmethod
    def clean_text(string):
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(string)
        updated_tokens = []
        stop_words = set(stopwords.words('english'))
        string.replace("``", "")
        for token in tokens:
            if "." in token and re.search(r"[A-Za-z].[A-Za-z]", token):
                updated_tokens.append(token.replace(".", " "))
            else:
                updated_tokens.append(token)
        return " ".join([lemmatizer.lemmatize(t.lower()) for t in updated_tokens if
                         t.lower() not in stop_words and t.lower() not in punctuation])

    @staticmethod
    def pre_process_spacy(string):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(string)
        clean = " ".join([token.lemma_.lower() for token in doc if token.pos_ != "PUNCT" and token.is_stop is False])
        ents = [e.text for e in nlp(clean).ents if e.label_ not in ('DATE', 'TIME', 'ORDINAL', 'CARDINAL')]
        # tokens = [e.lemma_.lower() for e in list(zip(*ents))[0] if e.pos_ != "PUNCT" and e.is_stop is False]

        return ents

        # spacy.displacy.render(doc, jupyter=True, style='ent')

        # return clean_tokens

    @staticmethod
    def get_extreme_sentiment_tokens(word_tokenized_article):
        terms = list(bigrams(word_tokenized_article)) + \
                list(trigrams(word_tokenized_article))

        extreme_tokens = []
        for term in terms:
            if type(term) != str:
                term = " ".join(term)
            blob = TextBlob(term)
            if abs(blob.sentiment.polarity) > 0.7 or blob.sentiment.subjectivity > 0.8:
                extreme_tokens.append(term)

        return extreme_tokens

    @staticmethod
    def get_polarity_score_article(clean_article):
        blob = TextBlob(clean_article)
        return blob.sentiment.polarity

    @staticmethod
    def most_frequently_occurring_keyword(string):
        keywords = ["middle east", "israel", "gaza", "hamas", "palestin", "syria", "turk", "egypt",
                    "iran", "saudi", "leban", "jordan", "qatar", "yemen"]

        freq = {key: string.count(key) for key in keywords}
        return max(freq, key=freq.get)

    @staticmethod
    def article_without_keywords(tokens):
        keywords = ["middle east", "israel", "gaza", "hamas", "palestin", "syria", "turk", "egypt",
                    "iran", "saudi", "leban", "jordan", "qatar", "yemen"]

        for x in keywords:
            for y in tokens:
                if x in y:
                    tokens.remove(y)

        return tokens

    @staticmethod
    def clean_df(df):
        df['clean_article'] = df.apply(lambda x: Processor.clean_text(x.text), axis=1)
        df['spacy_tokenized_article'] = df.apply(lambda x: Processor.pre_process_spacy(x.clean_article), axis=1)
        df['word_tokenized_article'] = df.apply(lambda x: Processor.pre_process(x.clean_article), axis=1)
        df['extreme_tokens'] = df.apply(lambda x: Processor.get_extreme_sentiment_tokens(x.word_tokenized_article),
                                        axis=1)
        df['clean_article'] = df.apply(lambda x: " ".join(x.word_tokenized_article), axis=1)
        df["article_summary"] = df.apply(lambda x: Processor.text_summarization(x.text), axis=1)
        df['article_polarity'] = df.apply(lambda x: Processor.get_polarity_score_article(x.clean_article), axis=1)
        df["source"] = df.apply(lambda x: "Fox"
                            if "foxnews.com" in x.url
                            else "Guardian"
                            if "guardian.com" in x.url
                            else None, axis=1
                        )
        df["most_frequently_occurring_keyword"] = \
            df.apply(lambda x: Processor.most_frequently_occurring_keyword(x.clean_article), axis=1)
        df["article_without_keywords"] = df.apply(lambda x:
                                                  Processor.article_without_keywords(x.word_tokenized_article), axis=1)
        # df['sentence_tokens'] = df.apply(lambda x: nltk.sent_tokenize(x.clean_article), axis=1)

        return df

    @staticmethod
    def process_df(df):
        return hstack([TfidfVectorizer(stop_words='english',
                                       min_df=10,
                                       max_df=50,
                                       ngram_range=(1, 4)
                                       ).fit_transform(df.clean_article),
                       TfidfVectorizer(tokenizer=lambda x: x,
                                       preprocessor=lambda x: x,
                                       token_pattern=None,
                                       min_df=10,
                                       max_df=50
                                       ).fit_transform(df.spacy_tokenized_article),
                       TfidfVectorizer(tokenizer=lambda x: x,
                                       preprocessor=lambda x: x,
                                       token_pattern=None,
                                       min_df=10,
                                       max_df=30
                                       ).fit_transform(df.extreme_tokens)
                       ]
                      )

    @staticmethod
    def most_frequent_keywords_by_cluster(df, cluster_type="kmeans", token_col="extreme_tokens"):

        dicts = {}
        for cluster in df[f"{cluster_type}_cluster"].unique():
            words_in_cluster = reduce(lambda x, y: x + y,
                                      df[df[f"{cluster_type}_cluster"] == cluster][token_col])
            frequencies = {}
            for word in words_in_cluster:
                if word in frequencies:
                    frequencies[word] += 1
                else:
                    frequencies[word] = 1

            converted_dict = dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))
            limited = {}
            for x in converted_dict.keys():
                if converted_dict[x] > 1 and len(limited.items()) < 10:
                    limited[x] = converted_dict[x]
            dicts[cluster] = limited

        return dicts

    @staticmethod
    def plot_clusters(data, labels):

        pca = PCA(n_components=2).fit_transform(data.todense())
        tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data.todense()))

        idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)

        plt.scatter(tsne[idx, 0], tsne[idx, 1], c=labels)
        plt.set_title('TSNE Cluster Plot')

    @staticmethod
    def create_co_occurrence_matrix(df):

        return df

    @staticmethod
    def generate_frequency_dictionary(df, column):

        freq = {}
        exclude = ["break key", "make sure", "far_beyond", "key story", "break key story", "right",
            "world", "government", "year", "even", "\'\'", "many", "state", "time", "biden", "last",
                   "life", "political", "international", "like", '\'s', "civilian", "week", "child",
                   "attack", "president", "latest book", "book fair", "update break key",
                   "key story telling", "email break key"]

        article_occurences = {}

        for index, article in df.iterrows():
            for token in article[column]:
                if token in exclude:
                    continue
                if token in freq:
                    freq[token] += 1
                else:
                    freq[token] = 1

        for index, article in df.iterrows():
            for token in article[column]:
                if token in article_occurences.keys() and index not in article_occurences[token]:
                    article_occurences[token].append(index)
                elif token not in article_occurences.keys():
                    article_occurences[token] = [index]

        article_occurences = {k: v for k, v in article_occurences.items() if 50 < len(v) < 300}

        return {k: v for k, v in freq.items() if k not in article_occurences.keys()}

    @staticmethod
    def most_frequent_keyword_wordcloud_by_source(df, source="Guardian", type_of_tokens="article_without_keywords"):
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        w1 = wordcloud.WordCloud(collocations=False) \
            .generate(' '.join(df[(df.source == source)]['most_frequently_occurring_keyword']))
        ax.imshow(w1, interpolation='bilinear')
        ax.axis("off")

        ax = fig.add_subplot(2, 1, 2)
        w2 = wordcloud.WordCloud().generate_from_frequencies(
            Processor.generate_frequency_dictionary(df[df.source == source], type_of_tokens))
        ax.imshow(w2, interpolation='bilinear')
        plt.axis("off")

        plt.show()

    @staticmethod
    def most_frequent_keyword_wordcloud_by_cluster(df, cluster_type="kmeans", column="article_without_keywords"):
        if not os.path.isdir(f"./static/word_cloud_by_cluster/{cluster_type}/{column}"):
            os.makedirs(f"./static/word_cloud_by_cluster/{cluster_type}/{column}")
        for x in df[f"{cluster_type}_cluster"].unique():
            fig = plt.figure(facecolor='#000000')
            ax = fig.add_subplot(2, 1, 1)
            w1 = wordcloud.WordCloud(collocations=False) \
                .generate(' '.join(df[(df[f"{cluster_type}_cluster"] == x)]['most_frequently_occurring_keyword']))
            ax.imshow(w1)
            ax.axis("off")

            ax = fig.add_subplot(2, 1, 2)
            freqs = Processor.generate_frequency_dictionary(df[df[f"{cluster_type}_cluster"] == x], column)

            freqs = {k: v for k, v in freqs.items()
                     if re.findall("middle east|israel|gaza|hamas|palestin|syria|turk|egypt|iran|saudi|leban|jordan|qatar|yemen", k) == []}

            freqs = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True))
            limited = {}
            for y in freqs.keys():
                if freqs[y] > 1 and len(limited.items()) < 50:
                    limited[y] = freqs[y]

            w2 = wordcloud.WordCloud().generate_from_frequencies(limited)
            ax.imshow(w2, interpolation='bilinear')
            plt.axis("off")

            if x == -1:
                plt.title("Noise points")

            plt.savefig(f"./static/word_cloud_by_cluster/{cluster_type}/{column}/cluster_{x}.png")

            plt.clf()
        return

    @staticmethod
    def create_source_chord_diagram(df, token_col="extreme_tokens"):
        # print(df[df.source == "Fox"].columns)
        # fox_words = reduce(lambda x, y: x + y,
        #                           df[df.source == "Fox"][token_col])
        # guardian_words = reduce(lambda x, y: x + y,
        #                    df[df.source == "Guardian"][token_col])
        #
        # unique_fox = set(fox_words) - set(guardian_words)
        # unique_guardian = set(guardian_words) - set(fox_words)
        #
        # fox_list = [x for x in fox_words if x in unique_fox]
        # guardian_list = [x for x in guardian_words if x in unique_guardian]
        #
        # print(unique_fox)
        #
        # select_cols = fox_list + guardian_list + ["source", "count", token_col]

        fox_df = df[df.source == "Fox"]
        guardian_df = df[df.source == "Guardian"]

        df_source_keyword = fox_df.explode(token_col).groupby(by=["source", token_col]) \
            .count()["dbscan_cluster"].rename("count")\
            .reset_index().sort_values(by="count", ascending=False).head(30)

        df_source_keyword = pd.concat([df_source_keyword, guardian_df.explode(token_col).groupby(by=["source", token_col]) \
            .count()["dbscan_cluster"].rename("count").reset_index()
                                      .sort_values(by="count", ascending=False).head(30)])

        df_source_keyword = df_source_keyword.rename(columns={token_col: "target"})

        source_keyword_combinations = list(set(df_source_keyword["source"].unique().tolist() +
                                               df_source_keyword["target"].unique().tolist()))

        dataset = hv.Dataset(pd.DataFrame(source_keyword_combinations, columns=["source"]))
        hv.extension('bokeh')
        hv.output(size=300)
        chord = Chord((df_source_keyword, dataset))
        chord.opts(
            opts.Chord(labels="source", edge_color=dim('source').str()))

        return hv.render(chord)

        #show(hv.render(chord))

    @staticmethod
    def create_word_clouds(df):
        freq_fox = {}
        freq_guardian = {}

        for index, article in df.iterrows():
            for token in article.spacy_tokenized_article:
                if "foxnews.com" in article.url:
                    if token in freq_fox:
                        freq_fox[token] += 1
                    else:
                        freq_fox[token] = 1
                else:
                    if token in freq_guardian:
                        freq_guardian[token] += 1
                    else:
                        freq_guardian[token] = 1

        fig = plt.figure()
        for i in range(2):
            ax = fig.add_subplot(2, 1, i + 1)
            w = wordcloud.WordCloud(collocations=False).generate_from_frequencies(freq_fox)
            ax.imshow(w, interpolation='bilinear')
            ax.axis("off")

            w = wordcloud.WordCloud(collocations=False).generate_from_frequencies(freq_guardian)
            ax.imshow(w, interpolation='bilinear')
            plt.axis("off")

        plt.show()

        # df.spacy_tokenized_article.apply()

    @staticmethod
    def run_clustering(tfidf, df):
        #km = KMeans(n_clusters=6, random_state=80)
        km = KMeans(n_clusters=6)
        clusters = km.fit(tfidf)
        df["kmeans_cluster"] = clusters.predict(Processor.process_df(df))

        dbscan = DBSCAN(eps=1.65)
        clusters = dbscan.fit(tfidf)

        df["dbscan_cluster"] = clusters.labels_

        no_clusters = len(np.unique(clusters.labels_))
        no_noise = np.sum(np.array(clusters.labels_) == -1, axis=0)

        print('Estimated no. of clusters: %d' % no_clusters)
        print('Estimated no. of noise points: %d' % no_noise)

        #print(clusters.labels_)

        # df["predicted_cluster"] = clusters.predict(Processor.process_df(df))
        return df

        # Processor.plot_clusters(tfidf, clusters)

        # dbscan = DBSCAN(eps=0.5, min_samples=5, algorithm='ball_tree', metric='minkowski', leaf_size=90, p=2)
        # dbscan.fit(tfidf)
        #
        # cluster_labels = dbscan.labels_
        # coords = tfidf.toarray()
        #
        # no_clusters = len(np.unique(cluster_labels))
        # no_noise = np.sum(np.array(cluster_labels) == -1, axis=0)

        # print('Estimated no. of clusters: %d' % no_clusters)
        # print('Estimated no. of noise points: %d' % no_noise)
        # print(cluster_labels)

    # df['bigrams'] = df.apply(lambda x: bigrams(x.tokenized_article), axis = 1)
    # df['tokenized_article'] = df.apply(lambda row: nltk.word_tokenize(row['Article']), axis=1)
    @staticmethod
    def wordcloud_by_most_frequent_keyword(df, column):
        if not os.path.isdir(f"./output/word_cloud_by_most_frequent_keyword/{column}"):
            os.makedirs(f"./output/word_cloud_by_most_frequent_keyword/{column}")
            
        figures = []

        for i in df.most_frequently_occurring_keyword.unique():
            figure = plt.figure()
            ax = figure.add_subplot(2, 1, 1)
            freqs = Processor.generate_frequency_dictionary(df[df.most_frequently_occurring_keyword == i], column)

            freqs = {k: v for k, v in freqs.items()
                     if re.findall(
                    "middle east|israel|gaza|hamas|palestin|syria|turk|egypt|iran|saudi|leban|jordan|qatar|yemen",
                    k) == []}

            freqs = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True))
            limited = {}
            for x in freqs.keys():
                if freqs[x] > 1 and len(limited.items()) < 50:
                    limited[x] = freqs[x]

            w2 = wordcloud.WordCloud().generate_from_frequencies(limited)

            inset = figure.add_axes([0.15, 0.55, .15, .15])
            inset.imshow(w2, interpolation='bilinear')
            inset.set_title(f"Words Associated with \'{i}\'")
            figure.axis("off")

            if x == -1:
                inset.set_title("Noise points")

            figure.show()

            figure.savefig(f"./output/word_cloud_by_most_frequent_keyword/{column}/{i}.png")

            html_str = mpld3.fig_to_html(figure)
            Html_file = open("project.html", "w")
            Html_file.write(html_str)
            Html_file.close()

            figures.append(plt.gcf())

            plt.clf()

    @staticmethod
    def text_summarization(string):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(string)

        keywords = []
        pos_tags = ["PROPN", "ADJ", "NOUN", "VERB"]

        for token in doc:
            if token.is_stop or token.text in punctuation:
                continue
            if token.pos_ in pos_tags:
                keywords.append(token.text)

        freq_word = Counter(keywords)

        sent_strength = {}
        for sent in doc.sents:
            for word in sent:
                if word.text in freq_word.keys():
                    if sent in sent_strength.keys():
                        sent_strength[sent] += freq_word[word.text]
                    else:
                        sent_strength[sent] = freq_word[word.text]

        return " ".join([w.text for w in nlargest(3, sent_strength, key=sent_strength.get)])

    @staticmethod
    def summaries_of_articles_containing_token(df, token):
        return df[df["clean_article"].str.contains(token)].article_summary
