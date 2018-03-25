import warnings
from asl_data import SinglesData

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses    
    for i in range(test_set.num_items):
        test_word_logL_dict = {}
        for word in list(models.keys()):
            X, lengths = test_set.get_all_Xlengths()[i]
            try:
                logL = models[word].score(X, lengths)
            except:
                logL = float('-inf')
            test_word_logL_dict[word] = logL
        probabilities.append(test_word_logL_dict)
        
        guesses.append(sorted(test_word_logL_dict.items(), key = lambda asd: asd[1], reverse = True)[0][0])
    
    return probabilities, guesses

