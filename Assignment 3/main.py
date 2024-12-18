# CS 471 AI - Assignment 3
# Implement the Naive Bayes classification method
# By: Dax Taraleskof and Shane Dyrdahl
# 10/03/2024

from typing import Dict, List, Tuple
from dataclasses import dataclass
from math import prod
from csv import reader

@dataclass
class Result:
    sentence: str
    prior: float
    conditional: Dict[str, float]
    posterior: float

def listify_csv(file_name: str) -> List[Tuple[str, str]]:
    """
    Attempts to open csv file with two columns: class,data
    and then returns it as a List of Tuples for each row.
    """
    with open(file_name) as f:
        csv_reader = reader(f)

        # skip the header objects
        next(csv_reader)

        contents = []
        for row in csv_reader:
            contents.append(row)

        return contents

def categorize(
    data: List[Tuple[str, str]],
    training_amount: int
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Returns a Dictionary of every word in the data list after it has been
    classified as Ham or Spam.
    """
    spam, ham = {}, {}
    for idx in range(training_amount):
        row = data[idx]
        words = row[1].rstrip().split(" ")
        for w in words:
            w = w.lower().strip(".")
            if row[0] == "ham":
                if w not in ham:
                    ham[w] = 1
                else:
                    ham[w] += 1
            elif row[0] == "spam":
                if w not in spam:
                    spam[w] = 1
                else:
                    spam[w] += 1

    return spam, ham

def probability_prior(
    data: List[Tuple[str, str]],
    data_num: int
) -> Tuple[float, float]:
    """
    Compute the prior probability for {data_num} amount of rows and returns
    two float values, the first being for SPAM and the second being for HAM.
    """
    total, spam_total, ham_total = 0, 0, 0
    for row in data:
        if data_num - total == 0: break
        if row[0] == "spam":
            spam_total += 1
        if row[0] == "ham":
            ham_total += 1
        total += 1

    return spam_total / total, ham_total / total

def probability_conditional(
        sentence: str,
        unique_words: int,
        occurence_per_word: Dict[str, int],
) -> Dict[str, float]:
    """Calculates the conditional probability for each word in a sentence and applies Laplace Smoothing."""
    total = sum(occurence_per_word.values())
    laplace_smoothing = total + unique_words

    conditional = {}
    words = sentence.split(" ")
    for w in words:
        if w.isspace():
            continue
        w = w.lower()
        if w not in occurence_per_word:
            conditional[w] = 1 / laplace_smoothing
        if w in occurence_per_word:
            conditional[w] = (occurence_per_word[w] + 1) / laplace_smoothing

    return conditional

def probability_posterior(
    prior_probability: float,
    conditional_probability: Dict[str, float],
) -> float:
    """Calculates the posterior probabilities given a set of conditional and prior values for both spam and ham."""
    return prod(conditional_probability.values()) * prior_probability

def calculate_probabilities(
        sentence: str,
        split: int,
        training_data: List[Tuple[str, str]],
) -> Tuple[Result, Result]:
    """
    Calculates the prior, conditional, and posterior probabilities for the
    given sentence and return a tuple of Results for ham and spam.
    """
    # Task 1. Load the dataset and split into training and testing sets (first 20 into training and the rest into testing)
    spam_training, ham_training = categorize(training_data, split) # split=20
    # Union of the set of keys from both training dictionaries. This filters any duplicates and returns the unique word count.
    unique_word_count = len(set(spam_training.keys() | ham_training.keys()))
    # Task 2. Calculate the prior probabilities: P(spam) and P(ham) (2 points)
    p_spam_prior, p_ham_prior = probability_prior(training_data, split)
    # Task 3. Calculate conditional probabilites: P(sentence|ham) and P(sentence|spam) (2 points)
    p_ham_conditional = probability_conditional(sentence, unique_word_count, ham_training)
    p_spam_conditional = probability_conditional(sentence, unique_word_count, spam_training)
    # Task 4. Compute the posterior probabilities: P(sentence|ham) and P(sentence|ham) (2 points)
    # The resulting float value is the posterior probability P(ham|sentence) or P(spam|sentence) for the given sentence.
    p_ham_posterior = probability_posterior(p_ham_prior, p_ham_conditional)
    p_spam_posterior = probability_posterior(p_spam_prior, p_spam_conditional)
    # Returns all of the computed probabilities as a python dataclass called 'Result', which is defined at the top of this file.
    return Result(sentence, p_spam_prior, p_spam_conditional, p_spam_posterior), Result(sentence, p_ham_prior, p_ham_conditional, p_ham_posterior)

def calculate_accuracy(
        testing_set: List[Tuple[str, str]],
        results: List[Tuple[Result, Result]]
) -> float:
    """
    Compares actual results from the results list to the original testing_set and calculates
    the accuracy of the Naive Bayes model.
    """
    correct_predictions = 0
    for idx, (classification, sentence) in enumerate(testing_set):
        spam_result, ham_result = results[idx]

        predicted_label = "spam" if spam_result.posterior > ham_result.posterior else "ham"

        if predicted_label == classification:
            correct_predictions += 1

    accuracy = correct_predictions / len(testing_set)
    return accuracy

def test_case(): # test case from the lecture slides, it produces identical output
    test_training_amount = 5

    try:
        contents_test = listify_csv("SpamDetectionTest.csv")
    except Exception as e:
        print("Please import SpamDetectionTest.csv into google colab")
        print("This was a custom .csv file that was made out of the lecture example")

    test_sentence_negative = "I hated the poor acting"
    test_sentence_positive = "I loved the great acting"

    test_sentence = test_sentence_negative

    negative_test_result, positive_test_result = calculate_probabilities(
        sentence=test_sentence_negative,
        split=test_training_amount,
        training_data=contents_test,
    )

    test_class_result = "negative" if negative_test_result.posterior > positive_test_result.posterior else "positive"
    print(f"'{test_sentence}' is classified as {test_class_result}")
    print(negative_test_result)
    print(positive_test_result)

show_probabilities = False

if __name__ == "__main__":
    #print("\nProblem from class:")
    #test_case()

    training_amount = 20
    contents = listify_csv("SpamDetection.csv")
    # Task 1. Split into testing set (training set is done in calculate_probabilities as the argument 'split')
    testing_set = contents[training_amount:]
    print("\nTesting Set:", testing_set, len(testing_set))

    print("\nTest Set from Assignment:")
    testing_set_results = []
    for classification, sentence in testing_set:
        spam_result, ham_result = calculate_probabilities(
            sentence=sentence,
            split=training_amount,
            training_data=contents,
        )

        if show_probabilities:
            print(f"\nPrior Probability of Spam: {spam_result.prior}")
            print(f"Prior Probability of Ham: {ham_result.prior}")

            print(f"\nConditional Probability of Spam: {spam_result.conditional}")
            print(f"Conditional Probability of Ham: {ham_result.conditional}")

            print(f"\nPosterior Probability of Spam: {spam_result.posterior}")
            print(f"Posterior Probability of Ham: {ham_result.posterior}")

        # Task 5. For each sentence in the test set: (2 points)
        # Display the sentence
        # Print the posterior probability of a sentence belonging to spam or ham
        # Display the class (spam or ham)
        actual_class = "spam" if spam_result.posterior > ham_result.posterior else "ham"
        print(f"\nSentence: '{sentence}'")
        print(f"Posterior Probabilities: SPAM={spam_result.posterior} | HAM= {ham_result.posterior}")
        print(f"Classification: ACTUAL={actual_class.upper()} | EXPECTED={classification.upper()}")

        # Gather all of the results in a list to calculate the accuracy of the model.
        testing_set_results.append((spam_result, ham_result))

    training_set_results = []
    for classification, sentence in contents[:training_amount]:
        spam_result_train, ham_result_train = calculate_probabilities(
            sentence=sentence,
            split=training_amount,
            training_data=contents,
        )
        training_set_results.append((spam_result_train, ham_result_train))

    # Report the train and test set accuracy (1 point)
    # Accuracy = no. of sentences correctly predicted by model / total sentences
    accuracy_of_model_training = calculate_accuracy(contents[:training_amount], training_set_results)
    print(f"\nTraining set accuracy : {accuracy_of_model_training * 100:.2f}%")

    accuracy_of_model_test = calculate_accuracy(testing_set, testing_set_results)
    print(f"\nTesting set accuracy : {accuracy_of_model_test * 100:.2f}%")
