import re

import numpy as np
from nltk.stem.porter import PorterStemmer

np.set_printoptions(precision=6)


def readFile(filename):
    """reads a file and returns its entire contents

    file_contents = readFile(filename) reads a file and returns its entire
    contents in file_contents
    :param filename:
    :return:
    """
    with open(filename, 'r') as fid:
        file_content = fid.read()
    return file_content


def getVocabList():
    """reads the fixed vocabulary list in vocab.txt and returns a
    cell array of the words

    vocabList = getVocabList() reads the fixed vocabulary list in vocab.txt
    and returns a cell array of the words in vocabList.
    :return:
    """
    ## Read the fixed vocabulary list
    with open('vocab.txt') as fid:
        vocabList = re.findall(r'\d+\t(\w+)', fid.read())
    return vocabList


def processEmail(email_contents):
    """preprocesses a the body of an email and returns a list of word_indices

    word_indices = processEmail(email_contents) preprocesses
    the body of an email and returns a list of indices of the
    words contained in the email.
    :param email_contents:
    :return:
    """
    # Load Vocabulary
    vocabList = getVocabList()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = email_contents.index('\n\n')
    # email_contents = email_contents[hdrstart + 2:]

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub(r'<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub(r'[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub(r'[$]+', 'dollar', email_contents)

    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('==== Processed Email ====\n')

    # Process file
    l = 0

    p = PorterStemmer()
    while email_contents:

        # Tokenize and also get rid of any punctuation
        str, email_contents = re.split(r'[ @\$/#\.-:&\*\+=\[\]\?!\(\)\{\},\'">_<;%\r\n]+', email_contents, maxsplit=1)

        # Remove any non alphanumeric characters
        str = re.sub(r'[^a-zA-Z0-9]', '', str)

        # Stem the word
        # (the porterStemmer sometimes has issues, so we use a try catch block)
        try:
            str = p.stem(str.strip())
        except:
            str = ''

        # Skip the word if it is too short
        if not str:
            continue

        # Look up the word in the dictionary and add to word_indices if
        # found
        if str in vocabList:
            word_indices.append(vocabList.index(str))

        # Print to screen, ensuring that the output lines are not too long
        if (l + len(str) + 1) > 78:
            print()
            l = 0
        print(str, end=' ')
        l = l + len(str) + 1

    # Print footer
    print('\n\n=========================')
    return word_indices


def emailFeatures(word_indices):
    """takes in a word_indices vector and produces a feature vector
    from the word indices

    x = emailFeatures(word_indices) takes in a word_indices vector and
    produces a feature vector from the word indices.
    :param word_indices:
    :return:
    """
    # Total number of words in the dictionary
    n = 1899

    x = np.zeros(n)
    for i in word_indices:
        x[i] = 1
    return x
