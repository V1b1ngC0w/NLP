import re
import contractions
import pandas as pd
from nltk.stem import LancasterStemmer


def preprocess(text:str)->str:

    def replace_contractions(text):
        """Replace contractions in string of text"""
        return contractions.fix(text)

    def remove_URL(sample):
        """Remove URLs from a sample string"""
        return re.sub(r"http\S+", "", sample)
    
    text = remove_URL(text)
    text = replace_contractions(text)

    return text

def normalise(text:str)->str:

    lancaster = LancasterStemmer()

    def lowerise(text):
        return text.lower()
    
    def remove_punctuation(text):
        # check if text is None or non string
        if pd.isna(text):
            return text
        return re.sub(r'[^\w\s]', '', str(text))
    
    def remove_common_words(text):
        common_words = ["the","our","and","a"]
        if pd.isna(text):
            return text
        text = " ".join([word for word in text.split() if word not in common_words])
        return text

    def stem_words(text):
        #! This function is kinda ass, ran it and it changed the words too much and too weird,
        #! will try to find a different library to stem words
        """
        Function that stems words in a list of words.
        E.g. starting -> start
        """
        if pd.isna(text):
            return text
        words = str(text).split()
        stemmed_words = [lancaster.stem(word) for word in words]

        return " ".join(stemmed_words)

    #TODO: Can add mnore functions here if needed, can do numbers->words

    
    text = lowerise(text)
    text = remove_punctuation(text)
    text = remove_common_words(text)
    #text = stem_words(text)

    return text
