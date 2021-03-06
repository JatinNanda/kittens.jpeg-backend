from archive_top_ngrams import get_ngrams_from_article_json
import csv
from datetime import datetime
import json
import os
import string
import numpy as np

DAY_MAP = {'Sun':0, 'Mon':1, 'Tue':2, 'Wed':3, 'Thu':4, 'Fri':5, 'Sat':6}
# call .map on dict to get array of features

def get_print_page(article):
    if 'print_page' not in article.keys():
        return 0
    print_page = article["print_page"]
    if (print_page is None or print_page == ''):
        return 0
    try:
        return int(print_page)
    except:
        #print "print_page error: ", print_page
        if "front" in print_page.lower():
            return 1
        return int(print_page.lower().strip(string.ascii_lowercase))


def get_num_multimedia(article):
    multimedia = article["multimedia"]
    return len(multimedia)

def get_headline_length(article):
    if ("main" in article["headline"]):
        main_headline = article["headline"]["main"].encode('utf-8').lower()
        return len(main_headline)
    return 0

def get_top_ngrams(ngram_csv, top_ngram_path="top-ngrams/"):
    top_ngrams = []
    with open(top_ngram_path + ngram_csv) as infile:
        reader = csv.reader(infile)
        for row in reader:
            ngram = row[0]
            top_ngrams.append(ngram)
    return top_ngrams

def get_ngram_vector(article, all_ngrams, ngram_csv, top_ngram_path="top-ngrams/"):
    vector = []
    if ("main" in article["headline"]):
        main_headline = article["headline"]["main"].lower()#.encode('utf-8').lower()
        # print main_headline
        ngrams_to_consider = get_top_ngrams(ngram_csv)
        for ngram in ngrams_to_consider:
            vector.append(main_headline.count(ngram))
        # print vector
    return vector

def get_keywords_vector(article, all_keywords):
    # the name is the type of keyword such as persons, subject, glocations
    article_keywords = [word["name"] for word in article["keywords"]]
    # returns the number of times each keyword in all_keywords shows up in the article keyword list
    vector = [article_keywords.count(key) for key in all_keywords]

    # vector = [1 if key in article_keywords else 0 for key in all_keywords]
    return vector

def get_pub_date(article):
    pub_date = article["pub_date"].split("T")[0]
    # example: "2015-09-18T07:57:43Z"
    # "Wed Jan 18 08:31:04 +0000 2017"
    datetime_object = datetime.strptime(pub_date, '%Y-%m-%d')

    return datetime_object.weekday()

def get_document_type_vector(article, all_doc_types):
    article_doc_type = article["document_type"]
    vector = [1 if section == article_doc_type else 0 for section in all_doc_types]
    return vector

def get_section_name_vector(article, all_section_names):
    if 'section_name' not in article.keys():
        vector = [0] * len(all_section_names)
    else:
        article_section = article["section_name"]
        vector = [1 if section == article_section else 0 for section in all_section_names]
    return vector

def get_news_desk_vector(article, all_news_desks):
    if 'news_desk' in article.keys():
        article_desk = article["news_desk"]
        vector = [1 if desk == article_desk else 0 for desk in all_news_desks]
    elif 'new_deks' in article.keys():
        # don't know where this is coming from, but it's a thing
        article_desk = article["new_desk"]
        vector = [1 if desk == article_desk else 0 for desk in all_news_desks]
    else:
        vector = [0] * len(all_news_desks)
    return vector

def get_subsection_name_vector(article, all_subsection_names):
    if 'subsection_name' not in article.keys():
        vector = [0] * len(all_subsection_names)
    else:
        article_subsection = article["subsection_name"]
        vector = [1 if subsection == article_subsection else 0 for subsection in all_subsection_names]
    return vector

def get_type_of_material_vector(article, all_types_of_materials):
    article_material = article["type_of_material"]
    vector = [1 if material == article_material else 0 for material in all_types_of_materials]
    return vector

def get_article_word_count(article):
    if article["word_count"]:
        return int(article["word_count"])
    else:
        return 0

def get_num_keywords(article):
    return len(article["keywords"])

def get_class(article):
    if 'was_tweeted' in article.keys():
        return int(article["was_tweeted"])
    else:
        return None


"""
FEATURES LIST
0: print_page
1: num_multimedia: len(multimedia)
2: headline_length: len(main_headline)
3: ngram_vector: main_headline: 50 ngrams and their occurrences (one-hot encoding)
4: keywords_vector: keywords.value (one-hot encoding)
5: pub_date: which day of the week (1-7)
6: document_type_vector: document_type (one-hot encoding)
7: section_name_vector: section_name (one-hot encoding)
8: news_desk_vector: news_desk (one-hot encoding)
9: subsection_name_vector: subsection_name (one-hot encoding)
10: type_of_material_vector: type_of_material (one-hot encoding)
11: article_word_count
12: num_keywords: len(keywords)
13: popularity: twitter: favorites/shares/etc.
"""

def get_instance_vector(article, all_ngrams, all_keywords, all_doc_types, all_section_names, all_news_desks, all_subsection_names, all_types_of_materials, ngram_csv):
    instance = []

    num_ngrams = len(all_ngrams)
    num_keywords = len(all_keywords)
    num_doc_types = len(all_doc_types)
    num_section_names = len(all_section_names)
    num_news_desks = len(all_news_desks)
    num_subsection_names = len(all_subsection_names)
    num_types_of_materials = len(all_types_of_materials)

    print_page = get_print_page(article) #0
    num_multimedia = get_num_multimedia(article) #1
    headline_length = get_headline_length(article) #2
    ngram_vector =  get_ngram_vector(article, all_ngrams, ngram_csv)#get_ngram_vector(article, all_ngrams) #3
    keywords_vector = get_keywords_vector(article, all_keywords) #4
    pub_date = get_pub_date(article) #5
    document_type_vector = get_document_type_vector(article, all_doc_types) #6
    section_name_vector = get_section_name_vector(article, all_section_names) #7
    news_desk_vector = get_news_desk_vector(article, all_news_desks) #8
    subsection_name_vector = get_subsection_name_vector(article, all_subsection_names) #9
    type_of_material_vector = get_type_of_material_vector(article, all_types_of_materials) #10
    article_word_count = get_article_word_count(article) #11
    num_keywords = get_num_keywords(article) #12
    was_tweeted = get_class(article) #13

    assert len(ngram_vector) == num_ngrams, " number of ngrams " + " actual " + str(len(ngram_vector)) + " expected " + str(num_ngrams)
    assert len(keywords_vector) == len(all_keywords), " number of keywords" + " actual " + str(len(keywords_vector)) + " expected " + str(num_keywords)
    assert len(document_type_vector) == num_doc_types, " number of document types " + " actual " + str(len(document_type_vector)) + " expected " + str(num_doc_types)
    assert len(section_name_vector) == num_section_names, " number of section names " + " actual " + str(len(section_name_vector)) + " expected " + str(num_section_names)
    assert len(news_desk_vector) == num_news_desks, " number of news desks " + " actual " + str(len(news_desk_vector)) + " expected " + str(num_news_desks)
    assert len(subsection_name_vector) == num_subsection_names, " number of subsection names " + " actual " + str(len(subsection_name_vector)) + " expected " + str(num_subsection_names)
    assert len(type_of_material_vector) == num_types_of_materials, " number of types of material " + " actual " + str(len(type_of_material_vector)) + " expected " + str(num_types_of_materials)

    instance = [print_page, num_multimedia, headline_length]
    instance.extend(ngram_vector)
    instance.extend(keywords_vector)
    instance.append(pub_date)
    instance.extend(document_type_vector)
    instance.extend(section_name_vector)
    instance.extend(news_desk_vector)
    instance.extend(subsection_name_vector)
    instance.extend(type_of_material_vector)
    instance.append(article_word_count)
    instance.append(num_keywords)

    return instance, was_tweeted

"""
all_ngrams, (in separate method above)
all_keywords,
all_doc_types,
all_section_names,
all_news_desks,
all_subsection_names,
all_types_of_materials
"""
def get_lists(all_articles_dict):

    all_keywords = [u'creative_works', u'glocations', u'organizations', u'persons', u'subject']

    all_doc_types = [u'article', u'blogpost', u'multimedia']

    all_section_names = [u'Arts', u'false', u'Sunday Review', u'The Upshot', u'Travel', u'Multimedia', u'Style', u'Sports', u'Health', u'Food', u'Crosswords & Games', u'Automobiles', u'World', u'NYT Now', u'Theater', u'Science', u'U.S.', u'Your Money', u'Movies', u'Magazine', u'Business Day', u'Opinion', u'Fashion & Style', u'Education', u'T Magazine', u'Books', u'Real Estate', u'Multimedia/Photos', u'N.Y. / Region', u'Technology']

    all_news_desks = [None, u'Arts', u'Arts / Art & Design', u'Arts / Television', u'Arts&Leisure', u'BookReview', u'Books', u'Books / Book Review', u'Business', u'Business Day', u'Business Day / International Business', u'Culture', u'Dining', u'EdLife', u'Editorial', u'Fashion & Style', u'Food', u'Foreign', u'Magazine', u'Metro', u'Metropolitan', u'Movies', u'Multimedia/Photos', u'N.Y. / Region', u'NYTNow', u'National', u'Obits', u'OpEd', u'Opinion', u'Opinion / Sunday Review', u'Politics', u'RealEstate', u'Science', u'Science / Space & Cosmos', u'Society', u'Sports', u'Sports / Skiing', u'Styles', u'Sunday Review', u'SundayBusiness', u'TStyle', u'The Upshot', u'Travel', u'U.S.', u'U.S. / Politics', u'Upshot', u'Weekend', u'World', u'World / Africa', u'World / Americas', u'World / Asia Pacific', u'World / Middle East']

    all_subsection_names = [None, u'Africa', u'Americas', u'Art & Design', u'Asia Pacific', u'Baseball', u'Book Review', u'College Basketball', u'College Football', u'Dance', u'DealBook', u'Dealbook', u'Economy', u'Education Life', u'Environment', u'Europe', u'International Arts', u'International Business', u'Media', u'Middle East', u'Music', u'Personal Tech', u'Politics', u'Pro Basketball', u'Pro Football', u'Skiing', u'Soccer', u'Space & Cosmos', u'Sunday Book Review', u'Sunday Review', u'Television', u'Tennis', u'Tony Awards', u'Weddings', u'false']

    all_types_of_materials = [u'An Analysis; News Analysis', u'Blog', u'Editorial', u'Interactive Feature', u'List', u'News', u'Obituary', u'Op-Ed', u'Op-Ed; Correction', u'Question', u'Review', u'Series', u'Special Report', u'Video', u'briefing']

    return all_keywords, all_doc_types, all_section_names, all_news_desks, all_subsection_names, all_types_of_materials

def get_all_instances(all_articles_dict, dataset_path="datasets/", ngram_path="/top-ngrams"):
    year = all_articles_dict[0]["pub_date"].split("-")[0]
    ngram_csv = year + "-TopNGrams.csv"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    all_ngrams = get_top_ngrams(ngram_csv)
    all_keywords, all_doc_types, all_section_names, all_news_desks, all_subsection_names, all_types_of_materials = get_lists(all_articles_dict)

    """
        instance = [print_page, num_multimedia, headline_length]
        instance.extend(ngram_vector)
        instance.extend(keywords_vector)
        instance.append(pub_date)
        instance.extend(document_type_vector)
        instance.extend(section_name_vector)
        instance.extend(news_desk_vector)
        instance.extend(subsection_name_vector)
        instance.extend(type_of_material_vector)
        instance.append(article_word_count)
        instance.append(num_keywords)
    """

    # prints out mapping of feature indicies for debugging reference
    # columns = ["print_page", "num_multimedia", "headline_length"]
    # columns.extend(all_ngrams)
    # columns.extend(all_keywords)
    # columns.append("pub_date")
    # columns.extend(all_doc_types)
    # columns.extend(all_section_names)
    # columns.extend(all_news_desks)
    # columns.extend(all_subsection_names)
    # columns.extend(all_types_of_materials)
    # columns.append("article_word_count")
    # columns.append("num_keywords")
    # for c, value in enumerate(columns, 0):
    #     print c, ": ", value


    dataset_name = ngram_csv.split("-")[0] + "-dataset.csv"
    instances = []
    labels = []

    with open(dataset_path + dataset_name, "w") as output:
        writer = csv.writer(output, delimiter=',')

        is_header = True
        for article in all_articles_dict:
            instance, label = get_instance_vector(article, all_ngrams, all_keywords, all_doc_types, all_section_names, all_news_desks, all_subsection_names, all_types_of_materials, ngram_csv)
            instances.append(instance)
            labels.append(label)
            if is_header:
                # column name is the column number from 0 to length of the data row including the label
                writer.writerow([i for i in xrange(len(instance) + 1)])
                is_header = False
            writer.writerow(instance + [label])

    instances = np.array(instances)
    labels = np.array(labels)
    return instances, labels

if __name__ == '__main__':
    ngram_csv = "2016_8-TopNGrams.csv"
    all_articles_dict = {}
    with open("sample_article.json", "r") as articlesjson:
        all_articles_dict = json.load(articlesjson)
    get_all_instances(all_articles_dict)
