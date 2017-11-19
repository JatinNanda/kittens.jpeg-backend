"""
uses python3
"""
import string
# import nltk
import csv
# from nltk.util import ngrams
from collections import Counter
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pprint
import operator
from read_archives import get_headlines
import os
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

pp = pprint.PrettyPrinter(indent=4)

def get_top_k_ngrams(archive, k, frequency_path = "ngrams/ngram-partials/"):
    if not os.path.exists(frequency_path):
        os.makedirs(frequency_path)
    input_name = frequency_path + archive + '-Headline-grams_frequencies.json'
    with open(input_name, 'r') as f:
        data = json.load(f)
        sorted_by_frequency = sorted(data.items(), key=operator.itemgetter(1), reverse=True)
        result = []
        for i in range(k):
            result.append(sorted_by_frequency[i])#[0])
        return result

def get_ngrams(archive, headline_path="headlines/", ngram_path="ngrams/"):
    if not os.path.exists(headline_path):
        os.makedirs(headline_path)
    if not os.path.exists(ngram_path):
        os.makedirs(ngram_path)
    stop_words = ENGLISH_STOP_WORDS.union(set(['8217', '8216']))
    vectorizer = CountVectorizer(stop_words=stop_words, lowercase=True, ngram_range=(1,3), token_pattern="\w{3,}")
    text = []

    archive_path = headline_path + archive
    with open(archive_path + '-Headline.txt', "r") as f:
        text = f.readlines()

    vectorized_corpus = vectorizer.fit_transform(text)
    words = [str(i) for i in vectorizer.get_feature_names()]
    counts = [int(i) for i in np.asarray(vectorized_corpus.sum(axis=0)).ravel()]

    result = zip(words, counts)
    result = dict(result)

    output_name = ngram_path + archive + "-Headline-grams.json"

    with open(output_name, 'w') as outfile:
        json.dump(result, outfile, indent=4)

    print(output_name)

def get_ngram_frequencies(archive, ngram_path="ngrams/", frequency_path="ngrams/ngram-partials/"):
    if not os.path.exists(ngram_path):
        os.makedirs(ngram_path)
    if not os.path.exists(frequency_path):
        os.makedirs(frequency_path)
    input_name = ngram_path + archive + '-Headline-grams.json'
    with open(input_name, 'r') as f:
        data = json.load(f)
        all_occurrences = data.values()
        max_raw_count = max(all_occurrences)
        result = {}
        for word in data.keys():
            occurrences = data[word]
            freq = float(occurrences) / float(max_raw_count)
            result[word] = freq

        output_name = frequency_path + archive + '-Headline-grams_frequencies.json'

        with open(output_name, 'w') as outfile:
            json.dump(result, outfile, indent=4)

        print(output_name)

def make_dict_keys_strings(dictionary):
    keys = dictionary.keys()
    keys_str = [str(key) for key in keys] # convert to strings

def get_ngrams_from_article_json(all_articles_dict, top_ngram_path="top-ngrams/"):
    if not os.path.exists(top_ngram_path):
        os.makedirs(top_ngram_path)

    archivefolder = "raw-archives/"
    year = all_articles_dict[0]['pub_date'].split('-')[0]#all_articles_dict[0]["created_at"].split(" ")[-1]
    get_headlines(all_articles_dict, year)

    archive_name = year
    get_ngrams(archive_name)
    get_ngram_frequencies(archive_name)
    result = get_top_k_ngrams(archive_name, 50)

    with open(top_ngram_path + archive_name + "-TopNGrams.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(result)

    # for archive in archivelist:
    #     archive_name = archive.split(".json")[0]
    #     get_ngrams(archive_name)
    #     get_ngram_frequencies(archive_name)
    #     result = get_top_k_ngrams(archive_name, 50)
    #
    #     with open(top_ngram_path + archive_name + "-TopNGrams.csv", "w") as f:
    #         writer = csv.writer(f)
    #         writer.writerows(result)

if __name__ == '__main__':
    archivelist = ["2016_8.json", "2015_8.json", "2014_8.json", "2013_8.json", "2012_8.json", "2011_8.json", "2010_8.json", "2009_8.json", "2008_8.json"] #["archive2017.json"]
    get_ngrams_from_article_json(archivelist)
