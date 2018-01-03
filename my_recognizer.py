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

    # List of dictionaries, each key is a word, value is Log L
    probabilities = []

    # List of the best guess words ordered by the test set word_id
    guesses = []

    x_lengths = test_set.get_all_Xlengths()
    for X, lengths in x_lengths.values():
        best_guess = None
        best_score = float('-inf')
        log_l = {}

        # Iterate through each word, using every given trained model
        for word, model in models.items():
            try:
                # Calculate log L for word and model
                score = model.score(X, lengths)

                # Save result in dict, to be appended to probabilities
                log_l[word] = score

                # Check for better score / greater word likelihood
                if score > best_score:
                    best_guess = word
                    best_score = score
            except:
                # Word cannot be processed
                log_l[word] = float('-inf')

        probabilities.append(log_l)
        guesses.append(best_guess)

    return probabilities, guesses

