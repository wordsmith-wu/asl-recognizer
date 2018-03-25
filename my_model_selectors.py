import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        n_components_range = range(self.min_n_components, self.max_n_components + 1)
        lowest_BIC = float('inf')
        best_model = self.base_model(self.min_n_components)
        for num_compoments in n_components_range:
            model = self.base_model(num_compoments)
            try:
                logL = model.score(self.X, self.lengths)
            except:
                break 
            p = num_compoments * (num_compoments - 1) + 2 * num_compoments * len(self.X[0]) 
            N = len(self.lengths) 
            BIC = -2 * logL + p * np.log(N)
            if BIC < lowest_BIC:
                lowest_BIC = BIC
                best_model = model
        
        return best_model
        

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        
        n_components_range = range(self.min_n_components, self.max_n_components + 1)
        highest_DIC = float('-inf')
        best_model = self.base_model(self.min_n_components)
        for num_compoments in n_components_range:
            model = self.base_model(num_compoments)
            try:
                logL = model.score(self.X, self.lengths)
            except:
                break
            DIC = logL - statistics.mean([model.score(self.hwords[word][0],self.hwords[word][1]) \
                                          for word in list(self.hwords.keys()) if word != self.this_word])
            if DIC > highest_DIC:
                highest_DIC = DIC
                best_model = model
        return best_model
    


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        n_components_range = range(self.min_n_components, self.max_n_components + 1)
        highest_CV = float('-inf')
        best_num_components = self.min_n_components
        
        for num_compoments in n_components_range:
            if len(self.sequences) < 3:
                return self.base_model(3)

            else:
                split_method = KFold()
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                    self.X, self.lengths = X_train, lengths_train
                    model = self.base_model(num_compoments)
                    try:
                        logL = model.score(X_test, lengths_test)
                    except:
                        logL = float('-inf') 
                    if logL > highest_CV:
                        highest_CV = logL
                        best_num_components = num_compoments
        self.X, self.lengths = self.hwords[self.this_word]
        best_model = self.base_model(best_num_components)
        return best_model

