import os
import numpy as np
import UnigramModel as ug
import BigramModel as bg

"""
Mixed Language Model (Unigram + Bigram)
@author Ben Gaudreau
@version 23 Feb 2024
"""

class MixedModel:
    """
    Construct a new mixed language model with the given corpus. The corpus
    should be a file path to a folder containing songs of a particular genre.
    This mixed language model is a combination of a unigram and bigram model.
    """
    def __init__(self, corpus):
        self.uniComponent = ug.UnigramModel(corpus)
        self.biComponent = bg.BigramModel(corpus)
        self.lam = 0.5
        return
    
    def test(self, inputText):
        return self.calculateProbability(inputText)
    
    def calculateProbability(self, inputText):
        genreProb = np.float64(0.0)
        uniProb = self.uniComponent.test(inputText)
        biProb = self.biComponent.test(inputText)
        genreProb = (self.lam*uniProb) + ((1-self.lam)*biProb)
        return genreProb

def buildModels(directory):
    # training data directory
    models = {}
    # build unigram models for each genre
    for genre in os.listdir(directory):
        songs = os.path.join(directory, directory + genre + "\\")
        model = MixedModel(songs)
        models.update({genre:model})
    return models

def testModels(models, inputText):
    max = 0
    result = ""
    for genre in models:
        genreProb = models[genre].test(inputText)
        if (genreProb > max):
            result = genre
            max = genreProb
    return result