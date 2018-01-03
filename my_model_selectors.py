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

        best_model = None
        best_bic = float('inf')

        # Calculate BIC score for n between self.min_n_components and self.max_n_components
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)

                log_l = model.score(self.X, self.lengths)
                p = n ** 2 + 2*n * model.n_features - 1
                bic_score = -2*log_l + p*math.log(n)

                # BIC Score: Lower is better
                if bic_score < best_bic:
                    best_model = model
                    best_bic = bic_score
            except:
                pass

        return best_model if best_model else self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_dic = float('-inf')
        log_l_all = []

        try:
            for n in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(n)
                log_l_all.append(model.score(self.X, self.lengths))

            for s in log_l_all:

                    # DIC second term = 1/(M-1)SUM(log(P(X(all but i))
                    dic_second_term = (sum(log_l_all) - s) / (len(log_l_all) - 1)

                    # DIC = log(P(X(i)) - second term
                    dic_score = s - dic_second_term

                    # DIC Score: Higher is better
                    if dic_score > best_dic:
                        best_model = model
                        best_dic = dic_score
        except:
            pass
        return best_model if best_model else self.base_model(self.n_constant)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_num_states = None
        best_cv = float('-inf')
        n_splits = 3
        split_method = KFold(n_splits=n_splits)

        for n in range(self.min_n_components, self.max_n_components + 1):
            fold_logl = []
            try:
                if len(self.sequences) < n_splits:
                    break

                # Break word sequences into folds, keep track of log Likelihood for each
                for train_idx, test_idx in split_method.split(self.sequences):

                    # Training data
                    train_x, train_length = combine_sequences(train_idx, self.sequences)

                    # Testing data in the fold
                    fold_x, fold_length = combine_sequences(test_idx, self.sequences)

                    # Train model on training data
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(train_x, train_length)

                    # Calculate score for fold, add to list of fold log L scores
                    logl = model.score(fold_x, fold_length)
                    fold_logl.append(logl)

                if len(fold_logl) > 0:
                    # CV Score = average logL of cross-validation folds
                    cv_score = np.mean(fold_logl)
                else:
                    # Unable to split into enough k-folds
                    cv_score = float("-inf")

                # CV Score: Higher is better
                if cv_score > best_cv:
                    best_cv = cv_score
                    best_num_states = n
            except:
                pass
        return self.base_model(best_num_states) if best_num_states is not None else self.base_model(self.n_constant)
