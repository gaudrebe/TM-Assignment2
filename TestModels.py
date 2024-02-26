import UnigramModel as ug
import BigramModel as bg
import MixedModel as mm

"""
Main method to run tests on language models.
@author Ben Gaudreau
@version 26 Feb 2024
"""

"""
Prepare a list of tests given a .tsv file of the format:
ID  Text    Genre
"""
def buildTests(testFile):
    testList = []
    with open(testFile, "r") as rfile:
        # skip header
        next(rfile)
        tests = rfile.readlines()
        for line in tests:
            currLine = line.split("\t")
            testID = currLine[0].strip()
            testText = currLine[1].strip()
            testValue = currLine[2].strip()
            testList.append((testID, testText, testValue))
    return testList

"""
Given language models and an input query, calculate the probabilities for each
model to produce the query, returned as a tuple.
"""
def runTest(models, testText):
    unigrams = models[0]
    bigrams = models[1]
    mixeds = models[2]
    unigramResult = ug.testModels(unigrams, testText)
    bigramResult = bg.testModels(bigrams, testText)
    mixedResult = mm.testModels(mixeds, testText)
    return (unigramResult, bigramResult, mixedResult)

def main():
    corpus = ()     # insert link to root folder of training data
    testFile = ()   # insert link to test file (.tsv)
    # construct language models and counter variables
    unigrams = ug.buildModels(corpus)
    bigrams = bg.buildModels(corpus)
    mixeds = mm.buildModels(corpus)
    tests = buildTests(testFile)
    unigramHits = 0
    bigramHits = 0
    mixedHits = 0
    # iterate over test list and print results to console
    for test in tests:
        testResults = runTest((unigrams, bigrams, mixeds), test[1])
        if (testResults[0] == test[2]):
            unigramHits += 1
        if (testResults[1] == test[2]):
            bigramHits += 1
        if (testResults[2] == test[2]):
            mixedHits += 1
        print("Test #%s:\n" % (test[0]))
        print("Expected value: %s\n" % (test[2]))
        print("Unigram: %s\n" % (testResults[0]))
        print("Bigram: %s\n" % (testResults[1]))
        print("Mixed: %s\n\n" % (testResults[2]))
    print("Unigram performance: %d/%d correct\n" % (unigramHits, len(tests)))
    print("Bigram performance: %d/%d correct\n" % (bigramHits, len(tests)))
    print("Mixed performance: %d/%d correct\n" % (mixedHits, len(tests)))
    return

main()