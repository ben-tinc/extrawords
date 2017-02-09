#!/usr/bin/env python3

import os
import logging
import gensim
import re

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "trainingscorpus"))
RESULT_FILE = os.path.join(PROJECT_DIR, "models", "alltexts_cleared.model")

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class MySentences(object):
    """Helper class to give access to files sentence wise in
    a memory friendly way.
    """

    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        # Prepare regex to delete all punctuation
        punct = [
            ",", ".", ";", "!", "?", "-", '"', "'",
            "``", "''", ":", "(", ")", "–"
        ]
        rx = '[' + re.escape(''.join(punct)) + ']'
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                line = re.sub(rx, ' ', line.lower())
                yield line.split()


def train(source_dir):
    sentences = MySentences(source_dir)
    model = gensim.models.Word2Vec(
        sentences,
        min_count=10,       # how often needs a word appear in the whole corpus to be used
        size=200,           # size of NN layers: greater value requires more training data
                            # but might yield better results (100 is default)
        workers=4,          # parallelization
    )
    return model


def extractKeywords(kwfile):
    with open(kwfile, "r") as f:
        titleline = f.readline()
        result = {num.strip(): [] for num in titleline.split(",")}
        for line in f.readlines():
            for i, el in enumerate(line.split(",")):
                if el:
                    parts = el.split("/")
                    if len(parts) == 2:
                        word = parts[0].lower()
                        freq = 0
                        try:
                            freq = int(parts[1].lower())
                        except ValueError:
                            pass
                        result[str(i + 1)].append((word, freq))
                    else:
                        print(parts)
    return result


def findSimilarWords(model, kw, number=10):
    kw = sorted(kw, reverse=True, key=lambda tupel: tupel[1])
    kw = kw[:number]
    more = []
    for word in kw:
        try:
            similar = model.most_similar(positive=[word], topn=5)
            similar_words = [w for w, p in similar]
            similar_words.insert(0, word[0])
            more.append(similar_words)
        except KeyError as e:
            print(e)
    return more


def main():
    # model = train("./sources/TAZ-Texte/extaz")
    # model.save("models/shorttest.model")
    # model = train(SOURCE_DIR)
    # model.save(RESULT_FILE)
    model = gensim.models.Word2Vec.load(RESULT_FILE)
    kw = extractKeywords("Schluesselwoerter.csv")
    for key in kw.keys():
        more = findSimilarWords(model, kw[key])
        with open("similar_" + key + ".txt", "w") as f:
            for e in more:
                f.write(str(e) + "\n")


if __name__ == '__main__':
    main()
