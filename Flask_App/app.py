from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
from IPython.display import HTML, display
from collections import defaultdict
import os
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import re

stop_words = stopwords.words('russian') + ['И', 'и', 'К','За','к', 'за','а','но','в','на','под','А','Под', 'Но', 'В','На']
toktok = ToktokTokenizer()
app = Flask(__name__)

from pymystem3 import Mystem
stem = Mystem()

from math import log
import json
k1 = 2.0
b = 0.75

def score_BM25(n, qf, N, dl, avdl):
    K = compute_K(dl, avdl)
    IDF = log((N - n + 0.5) / (n + 0.5))
    frac = ((k1 + 1) * qf) / (K + qf)
    return IDF * frac


def compute_K(dl, avdl):
    return k1 * ((1-b) + b * (float(dl)/float(avdl)))
from collections import defaultdict

with open('inverse_inds.txt', 'r') as file:
    inverted_inds = json.load(file)

with open('lens.txt', 'r') as file:
    lens_of_texts = json.load(file)

# print(inverted_inds)
N = 1151
avdl = 691.1078260869565
# print(inverted_inds)

# lens_of_texts

def search(query):
    relevance_list = defaultdict(int)
    query = query.split()
    for word in query:
        if word not in stop_words:
            word = word.lower()
            q = stem.lemmatize(word)[0]
            if q in inverted_inds:
                docarray = inverted_inds[q]
            else:
                docarray = []
            for doc_name in docarray:
                # print(doc_name)
                n = len(inverted_inds[q])
                qf = inverted_inds[q][doc_name]
                if 'ipynb' not in doc_name:
                    text = open('corpus/' + doc_name, 'r')
                    read_txt = text.read()
                    index = score_BM25(n, qf, N, lens_of_texts[doc_name], avdl)
                    # print(index)
                    link = re.findall('@url(.+)', read_txt)[0]
                    title = re.findall('@ti(.+)', read_txt)[0]
                    #             print(link)
                    #             break
                    relevance_list[(link, title)] += index

    arr = sorted(relevance_list.items(), key=lambda x: x[1], reverse=True)[:10]
    # print(arr)
    return arr


@app.route('/')
def index():
    if request.args:
        query = request.args['query']
        ret = search(query)
        # title = ret[1]
        return render_template('index.html', links=ret, after_query = True)
    return render_template('index.html', links=[], after_query = False)


if __name__ == '__main__':
    app.run(debug=True)
