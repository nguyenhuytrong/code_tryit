import math
import string
import os
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                files[filename] = file.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(document.lower())
    return [word for word in tokens if word.isalnum() and word not in stop_words]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}
    total_documents = len(documents)

    for document in documents:
        for word in set(documents[document]):
            idfs[word] = idfs.get(word, 0) + 1

    for word in idfs:
        idfs[word] = math.log(total_documents / idfs[word])

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    scores = {}
    for filename, words in files.items():
        scores[filename] = sum(idfs[word] for word in query if word in words)

    return sorted(scores.keys(), key=lambda filename: scores[filename], reverse=True)[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    scores = {}
    for sentence, words in sentences.items():
        matching_words = [word for word in query if word in words]
        scores[sentence] = sum(idfs[word] for word in matching_words)

    sorted_sentences = sorted(scores.keys(), key=lambda sentence: (
        scores[sentence], query_term_density(sentence, query)), reverse=True)
    return sorted_sentences[:n]


def query_term_density(sentence, query):
    """
    Calculate query term density for a sentence.
    """
    sentence_words = word_tokenize(sentence.lower())
    return sum(1 for word in sentence_words if word in query) / len(sentence_words)


if __name__ == "__main__":
    main()
