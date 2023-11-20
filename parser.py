import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | S VP Conj VP
AP -> Adj | AP Adj
PP -> P | P NP
NP -> N | AP NP | NP PP | Det N | Det AP NP
VP -> V | V NP | VP P NP | VP NP P | V Adv | Adv VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase and the first phrase of the VP
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    words = nltk.word_tokenize(sentence.lower())
    return [word for word in words if any(c.isalpha() for c in word)]


# def np_chunk(tree):
#     """
#     Return a list of all noun phrase chunks in the sentence tree.
#     A noun phrase chunk is defined as a subtree of the sentence
#     whose label is "NP" that does not itself contain other
#     noun phrases as subtrees.
#     """
#     np_chunks = []

#     def is_noun_phrase_chunk(subtree):
#         return (
#             subtree.label() == 'NP' and
#             not any(t.label() == 'NP' for t in subtree.subtrees() if t != subtree)
#         )

#     for subtree in tree.subtrees(filter=is_noun_phrase_chunk):
#         np_chunks.append(subtree)

#     return np_chunks

def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as a subtree of the sentence
    whose label is "NP".
    """
    np_chunks = []

    def is_noun_phrase_chunk(subtree):
        return subtree.label() == 'NP'

    for subtree in tree.subtrees(filter=is_noun_phrase_chunk):
        np_chunks.append(subtree)

    return np_chunks


if __name__ == "__main__":
    main()
