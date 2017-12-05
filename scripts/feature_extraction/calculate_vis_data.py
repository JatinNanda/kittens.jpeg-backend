# from archive_top_ngrams import get_ngrams
from read_archives import get_stops
import json
import csv
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from string import punctuation

def get_ngrams(archive, headlines_list, min_ngram=1, max_ngram=3):

    stop_words = ENGLISH_STOP_WORDS.union(set(['8217', '8216']))
    vectorizer = CountVectorizer(stop_words=stop_words, lowercase=True, ngram_range=(min_ngram,max_ngram), token_pattern="\w{3,}")
    # text = []

    vectorized_corpus = vectorizer.fit_transform(headlines_list)
    words = [str(i) for i in vectorizer.get_feature_names()]
    counts = [int(i) for i in np.asarray(vectorized_corpus.sum(axis=0)).ravel()]

    result = zip(words, counts)
    result = dict(result)

    # output_name = ngram_path + archive + "-Headline-grams.json"
    #
    # with open(output_name, 'w') as outfile:
    #     json.dump(result, outfile, indent=4)
    #
    # print(output_name)
    return result

def clean_headline(headline, stop_phrases):
    for phrase in stop_phrases:
        try:
            headline = headline.replace(phrase, "").encode('utf-8')
        except:
            # print "COULD NOT PARSE: ", headline
            headline = ""
    return headline

def remove_stop_phrases_from_headlines(year):
    # if not os.path.exists("headlines/classifier_output/"):
    #     os.makedirs("headlines/classifier_output/")

    classifier_output_path = 'outputs/' + year + '-headlines.csv'
    stop_phrases = get_stops()

    clean_headlines = []

    with open(classifier_output_path, 'r') as infile:
        reader = csv.reader(infile)
        is_header = True
        for row in reader:
            if (is_header):
                is_header = False
            else:
                headline = clean_headline(row[1].lower(), stop_phrases)
                clean_headlines.append(headline)

    return clean_headlines#'\n'.join(clean_headlines)

def get_unigrams_from_popular_headlines(year, headlines_str):
    unigrams = get_ngrams(year, headlines_str, min_ngram=1, max_ngram=1)
    formatted_unigrams = {}
    for key in unigrams:
        # count = int(unigrams[key])
        formatted_unigrams[key] = {}
        formatted_unigrams[key]["year_count"] = 0#count
        formatted_unigrams[key]["times_tweeted"] = 0
        formatted_unigrams[key]["times_not_tweeted"] = 0
    return formatted_unigrams

def append_tweet_count(unigrams_object, year):
    classifier_output_path = 'outputs/' + year + '-headlines.csv'
    stop_phrases = get_stops()
    with open(classifier_output_path, 'r') as infile:
        reader = csv.reader(infile)
        is_header = True
        for row in reader:
            if (is_header):
                is_header = False
            else:
                # headline_words = set(row[0].translate(None, punctuation).lower().split(" "))
                ch = clean_headline(row[1].lower(), stop_phrases)
                # if " says " in ch:
                #     print ch
                headline_words = set(ch.translate(None, punctuation).split(" "))

                was_tweeted = int(float(row[3]))
                # print headline_words
                for word in headline_words:
                    if (word in unigrams_object):
                        if (was_tweeted == 1):
                            unigrams_object[word]["times_tweeted"] += 1
                        else:
                            unigrams_object[word]["times_not_tweeted"] += 1
                        unigrams_object[word]["year_count"] += 1
    return unigrams_object

if __name__ == '__main__':
    archive_list = []#['1861', '1900']
    for year in xrange(1851, 1899):
        archive_list.append(str(year))
    data = {}
    for year in archive_list:
        clean_headlines = remove_stop_phrases_from_headlines(year)
        unigrams = get_unigrams_from_popular_headlines(year, clean_headlines)
        tweets_and_counts_per_year = append_tweet_count(unigrams, year)
        print year, len(tweets_and_counts_per_year)
        data[year] = tweets_and_counts_per_year
    with open('vis_data.json', 'w') as output:
        json.dump(data, output, indent=4)
