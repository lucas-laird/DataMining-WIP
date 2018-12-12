import os
from nltk.corpus import stopwords
import re

def read_file(filename):
    stopWords = set(stopwords.words('english'))
    with open(filename, 'r',encoding = 'utf-8', errors = 'surrogateescape') as f:
        doc = f.read()
        reviews = doc.split('<EOR>')
    data = [];
    for r in reviews[:-1]:
        temp = re.sub(r'([^\s\w]|_)+', '', r)
        temp = temp.split()
        temp = list(filter(lambda a: a != '', temp))
        temp2 = []
        for t in temp:
            t = t.lower()
            if t not in stopWords:
                temp2.append(t)
        data.append(temp2)
    rating = reviews[len(reviews)-1]
    rating = rating[10:13]
    return(data,rating)

def read_dir(directory):
    files = os.listdir(directory)
    i = 0
    n = len(files)
    while i < n:
        filename = files[i]
        i = i+1
        if filename.endswith('.txt'):
            business_id = filename.split('.txt')[0]
            filename = directory + '/' + filename
            data,rating = read_file(filename)
            yield(data,business_id,rating)

def read_reviews(directory):
    files = os.listdir(directory)
    i = 0
    n = len(files)
    d = []
    while i < n:
        filename = files[i]
        i = i+1
        if filename.endswith('.txt'):
            filename = directory + '/' + filename
            temp = read_file(filename)
            for i in temp[0]:
                d.append(i)
    return(d)
