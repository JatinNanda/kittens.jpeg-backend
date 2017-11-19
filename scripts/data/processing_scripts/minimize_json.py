import json
from os import listdir
from os.path import isfile, join

mypath = "../nyt-files/archives-json/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and ".py" not in f and ".zip" not in f]

for f in onlyfiles:
    print f
    data = None
    with open(mypath + f, "rb") as infile:
        data = json.load(infile)
    with open(mypath + f, "wb") as outfile:
        json.dump(data, outfile)

