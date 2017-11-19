import json
import os
from os import listdir
from os.path import isfile, join
import sys

"""
input: list of paths to json files of article metadata
output: unique main headlines for that time period (month)
"""
def get_stops(year=None):
	mypath = "headline-stops/"
	if year == None:
		stop_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	else:
		stop_files = [year + ".txt"]

	phrases = set()
	for file_name in stop_files:
		path = mypath + file_name
		with open(path, "r") as stop_file:
			phrases = phrases.union(set([line.strip() for line in  stop_file.readlines()]))
	return phrases

def get_headlines(all_articles_dict, year):
	stop_phrases = get_stops()
	archive_name = year
	archive_headline_path = "headlines/" + archive_name + "-Headline.txt"
	temp_file = "headlines/temp.txt"

	with open(temp_file, "w") as outfile:
		for article in all_articles_dict:
			if ("main" in article["headline"]):
				main_headline = article["headline"]["main"].encode('utf-8').lower()
			if not any(phrase in main_headline for phrase in stop_phrases):
				outfile.write(main_headline + "\n")
			# for phrase in stop_phrases:
			# 	main_headline = main_headline.replace(phrase, "")
			outfile.write(main_headline + "\n")
	headlines = []
	with open(temp_file, "r") as headlinefile:
		headlines = headlinefile.readlines()
		headlines = sorted(headlines)
	with open(archive_headline_path, "w") as headlinefile:
		headlines = "".join(headlines)
		headlinefile.write(headlines)
	print "got headlines for: ", archive_headline_path
	#delete temp
	os.remove(temp_file)

# def get_headlines(archivefolder, archivelist):
#     if not os.path.exists(archivefolder):
#         os.makedirs(archivefolder)
#     stop_phrases = get_stops()
#     for archive in archivelist:
# 		archive_name = archive.split(".json")[0];
# 		archive_headline_path = "headlines/" + archive_name + "-Headline.txt"
# 		temp_file = "headlines/temp.txt"
# 		archive_json_path = archivefolder + archive
#
# 		with open(archive_json_path, "r") as myfile:
# 			data = json.load(myfile)
# 			data = data[0]
# 			for article in data["response"]["docs"]:
# 				if ("main" in article["headline"]):
# 					main_headline = article["headline"]["main"].encode('utf-8').lower()
# 					# print "main_headline", main_headline
# 					if not any(phrase in main_headline for phrase in stop_phrases):
# 						with open(temp_file, "a") as outfile:
# 							outfile.write(main_headline + "\n")
# 					else:
# 						print "IGNORED: ", main_headline
#
# 		headlines = []
# 		with open(temp_file, "r") as headlinefile:
# 			headlines = headlinefile.readlines()
# 		headlines = sorted(headlines)
# 		with open(archive_headline_path, "w") as headlinefile:
# 			headlines = "".join(headlines)
# 			headlinefile.write(headlines)
# 		print "got headlines for: ", archive_headline_path
# 		#delete temp
# 		os.remove(temp_file)
