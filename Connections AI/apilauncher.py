# TAMU datathon winner 2024 for connections AI challenge! We won 4 free electric scooters from this!

from flask import Flask, request, jsonify
import json
import argparse
import spacy
from functools import lru_cache
import numpy as np
import logging
from collections import defaultdict
import openai

app = Flask(__name__)
openai.api_key = "INSERT_KEY_HERE"
logging.basicConfig(level=logging.INFO) #simple logging


nlp = spacy.load('en_core_web_md')#loadin spacy

data1,data2 = {},{}

def load_data(file1, file2):
    global data1, data2
    try:
        with open(file1, encoding='utf-8') as f1:
            data1 = json.load(f1)
        # Convert all words in data1 to uppercase for consistency
        for category in data1:
            data1[category] = [word.upper() for word in data1[category]]
    except Exception as e:
        logging.error(f"Error loading data from {file1}: {e}")
        data1 = {}

    try:
        with open(file2, encoding='utf-8') as f2:
            data2 = json.load(f2)
        # Convert all words in data2 to uppercase for consistency
        for category in data2:
            data2[category] = [word.upper() for word in data2[category]]
    except Exception as e:
        logging.error(f"Error loading data from {file2}: {e}")
        data2 = {}

@lru_cache(maxsize=None)
def get_doc(text): #doc objects to improve performance
    return nlp(text.lower())

def cosine_similarity(a, b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)) # return the similarity by dot product over the vectors 

@app.route('/find_category', methods=['POST'])
def find_category():
    try:
        inputWords = request.json.get('words', [])
        if isinstance(inputWords,list)==False:
            return jsonify({"error": "Input must be a list of words."}), 400

        inputSet = set(word.upper() for word in inputWords)

        # remove existing guess words
        correct_groups = request.json.get('correctGroups', [])
        words_to_remove = set()
        for group in correct_groups:
            for word in group:
                words_to_remove.add(word.upper())
        inputSet -= words_to_remove

        #get the isOneAway and previousGuesses from request and convert to a set
        is_one_away = request.json.get('isOneAway', False)
        previous_guesses = request.json.get('previousGuesses', [])


        previous_guesses_sets = set(frozenset(map(str.upper, guess)) for guess in previous_guesses)#frozen set can be used for easier comparison

        # First, try with data1, exact_four_match_only=True
        matches = process_data(data1, inputSet, is_one_away, previous_guesses_sets, previous_guesses, exact_four_match_only=True)
        if matches:
            return jsonify({"matches": matches}), 200

        # If no matches found in data1(the previous game set), try with data2(the expanded data set.)
        #ensuring that palindrome, anagram and contains check is done before processing the final openai fallback call.
        additional_matches = check_palindromes_anagram_contains(inputSet, previous_guesses_sets)
        if additional_matches and (additional_matches not in previous_guesses):
            return jsonify({"matches": additional_matches}), 200

        matches = process_data(data2, inputSet, is_one_away, previous_guesses_sets, previous_guesses)
        if matches:
            return jsonify({"matches": matches}), 200

    

        # If no matches are found
        return jsonify({"error": "No matching category found with at least two words."}), 400

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

def process_data(data, input_set, is_one_away, previous_guesses_sets, previous_guesses, exact_four_match_only=False):

    local_input_set = input_set.copy() # usin a copy

    #can make this more efficient by caching somehow im sure. (however its already lightning fast if this function is called)
    categories = []
    for category, members in data.items():
        categories.append({
            "category": category,
            "words": set(word.upper() for word in members)
        })

    # If isOneAway is True, remove previous guess word from set
    if is_one_away and previous_guesses:
        last_guess = previous_guesses[-1]
        last_guess_set = set(word.upper() for word in last_guess)

        # Remove the last guess words from the input set
        local_input_set -= last_guess_set

        matching_categories = []
        for category in categories:
            # Find how many words from the last guess are in this category
            intersection = category["words"] & last_guess_set
            if len(intersection) == 3:
                # The incorrect word is the one not in the intersection
                incorrect_word = (last_guess_set - intersection).pop()
                # Remove the incorrect word from available words
                available_words = local_input_set - {incorrect_word}

                # Use the three correct words to predict the fourth word
                matched_words = list(intersection)
                predicted_word = predict_fourth_word(
                    category_name=category["category"],
                    category_words=category["words"],
                    matched_words=matched_words,
                    input_set=available_words,
                    previous_guesses_sets=previous_guesses_sets
                )
                if predicted_word:
                    # Append the predicted word
                    matched_words.append(predicted_word)
                    new_guess_set = frozenset(matched_words)
                    if new_guess_set not in previous_guesses_sets:
                        matching_categories.append({
                            "category": category["category"],
                            "matched_words": matched_words
                        })
                else:
                    # If no prediction, just return the matched words if they form a new guess
                    new_guess_set = frozenset(matched_words)
                    if new_guess_set not in previous_guesses_sets:
                        matching_categories.append({
                            "category": category["category"],
                            "matched_words": matched_words
                        })
        if matching_categories:
            return matching_categories
        else:
            # If no matching categories found, proceed as usual
            pass

    # Proceed with the normal matching process

    # First, attempt to find all categories with at least four matching words
    matching_categories = []
    for category in categories:
        intersection = category["words"] & local_input_set
        matched_words = list(intersection)

        if len(intersection) >= 4:
            # Select exactly four matched words
            selected_words = matched_words[:4]
            new_guess_set = frozenset(selected_words)
            if new_guess_set not in previous_guesses_sets:
                matching_categories.append({
                    "category": category["category"],
                    "matched_words": selected_words
                })

    if matching_categories:
        return matching_categories

    # If exact_four_match_only is True, return None
    if exact_four_match_only:
        return None

    # If no categories have four matches, attempt to find categories with exactly three matches
    predicted_categories = []
    for category in categories:
        intersection = category["words"] & local_input_set
        matched_words = list(intersection)

        if len(intersection) == 3:
            # Attempt to predict the 4th word from local_input_set using category name as a hint
            predicted_word = predict_fourth_word(
                category_name=category["category"],
                category_words=category["words"],
                matched_words=matched_words,
                input_set=local_input_set,
                previous_guesses_sets=previous_guesses_sets
            )
            if predicted_word:
                # Append the predicted word as the last item
                matched_words.append(predicted_word)
                new_guess_set = frozenset(matched_words)
                if new_guess_set not in previous_guesses_sets:
                    predicted_categories.append({
                        "category": category["category"],
                        "matched_words": matched_words
                    })

    if predicted_categories:
        return predicted_categories

    # If neither matches nor predictions are found, attempt to predict third and fourth words for categories with two matches
    additional_predicted_categories = []
    for category in categories:
        intersection = category["words"] & local_input_set
        matched_words = list(intersection)

        
            # Attempt to predict the 3rd and 4th words from local_input_set using category name as a hint
        predicted_words = predict_third_and_fourth_word(
                category_name=category["category"],
                category_words=category["words"],
                matched_words=matched_words,
                input_set=local_input_set,
                previous_guesses_sets=previous_guesses_sets
            )
       
        if predicted_words:
            return predicted_words
                # Append the predicted words as the last items
                # matched_words=[]
                # matched_words.extend(predicted_words)
                # new_guess_set = frozenset(matched_words)
                # if new_guess_set not in previous_guesses_sets:
                #     additional_predicted_categories.append({
                #         "category": category["category"],
                #         "matched_words": matched_words
                #     })

    if additional_predicted_categories:
        return additional_predicted_categories

    # If no matches found
    return None

def predict_fourth_word(category_name, category_words, matched_words, input_set, previous_guesses_sets):
    logging.info(f"PREDICTION CALLED: catname: {category_name} Matched Words: {matched_words}, Category Words: {category_words}, Input Set: {input_set}")
    """
    Predicts the fourth word for a category based on semantic similarity.
    The predicted word is from the input_set and not already in matched_words.
    It uses the matched words, remaining category words, and the category name as context.
    """
    # Words available for prediction: in input_set but not matched
    available_words =input_set-set(matched_words)

    if not available_words:
        return None

    # Get the remaining category words (excluding matched_words)
    remaining_category_words = category_words - set(matched_words)

    # Create context vectors
    context_docs = [get_doc(word) for word in matched_words]
    context_docs += [get_doc(word) for word in remaining_category_words]
    context_docs.append(get_doc(category_name))

    # Compute the average vector for the context
    context_vector = sum(doc.vector for doc in context_docs) / len(context_docs)

    # Compute similarities
    candidates = []
    for word in available_words:
        word_doc = get_doc(word)
        similarity = cosine_similarity(word_doc.vector, context_vector)
        # Form a new guess set
        new_guess_set = frozenset(matched_words + [word])
        if new_guess_set in previous_guesses_sets:
            continue  # Skip if this guess has already been made
        candidates.append((word, similarity))

    if not candidates:
        return None

    # Select the candidate with the highest similarity
    candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
    predicted_word, max_score = candidates_sorted[0]

    return predicted_word


def predict_third_and_fourth_word(category_name, category_words, matched_words, input_set, previous_guesses_sets):
    print('OPENAICALLEDLALALALALA-------------------------------------------------------------------------------')
    available_words = set(input_set) - set(matched_words)

    # System prompt
    system_prompt = (
        "You are an expert puzzle solver. You understand literature and you are well versed in wordplay. "
        "I want you to solve a daily word puzzle that finds commonalities between words.\n"
    )

    # User prompt
    user_prompt = (
        "Here is a word puzzle:\n"
        "- There are several words, which form groups of 4 words. Each group has some common theme that links the words.\n"
        "- You must use each word only once.\n"
        "- Each group of 4 words is linked together in some way.\n"
        "- Provide the next best group of 4 words that are linked together in some way.\n"
        "- Only output the list of 4 words, separated by commas, and nothing else.\n"
        "- Do not output any reasoning or explanation.\n"
        "- Do not include any words that have already been used.\n"
        "- Do not include any previous guesses.\n"
        "Here are the available words:\n" + ', '.join(available_words) + "\n"
    )

    # Add previous guesses to the prompt if any
    if previous_guesses_sets:
        user_prompt += "Previous guesses (do not repeat these):\n"
        for guess in previous_guesses_sets:
            user_prompt += '- ' + ', '.join(guess) + '\n'

    # Prepare the messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Call the OpenAI API (make sure your API key is set)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=50,
        temperature=0.7
    )

    assistant_reply = response['choices'][0]['message']['content']

    words_in_reply = [word.strip() for word in assistant_reply.split(',')]

    words_in_reply = [word for word in words_in_reply if word in available_words]
    words_in_reply = list(set(words_in_reply))

    if len(words_in_reply) != 4:
        return []

    previous_guesses_sets = [set(guess) for guess in previous_guesses_sets]
    if set(words_in_reply) in previous_guesses_sets:
        return []

    return words_in_reply


def check_palindromes_anagram_contains(input_set, previous_guesses_sets):
    """
    Checks for palindromes, anagram pairs, and words containing the same substring of length 2 or more.
    Returns a list of matched categories if found, else None.
    """
    # Convert input_set to list for indexing
    input_list = list(input_set)

    # Palindromes
    palindromes = [word for word in input_set if word == word[::-1]]
    if len(palindromes) >= 4:
        matched_words = palindromes[:4]
        new_guess_set = frozenset(matched_words)
        if new_guess_set not in previous_guesses_sets:
            return [{
                "category": "Palindromes",
                "matched_words": matched_words
            }]

    # Anagrams
    anagram_groups = defaultdict(list)
    for word in input_set:
        sorted_letters = ''.join(sorted(word))
        anagram_groups[sorted_letters].append(word)

    # Find all groups with at least two words
    anagram_pairs = [words for words in anagram_groups.values() if len(words) >= 2]

    # If we can find two such groups, get two words from each
    if len(anagram_pairs) >= 2:
        matched_words = []
        for group in anagram_pairs[:2]:
            matched_words.extend(group[:2])
        new_guess_set = frozenset(matched_words)
        if new_guess_set not in previous_guesses_sets:
            return [{
                "category": "Anagrams",
                "matched_words": matched_words
            }]

    # Contains same substring of length 2 or more
    substring_counts = defaultdict(list)
    for word in input_set:
        word_length = len(word)
        substrings = set()
        for length in range(3, word_length + 1):
            for i in range(word_length - length + 1):
                substring = word[i:i+length]
                substrings.add(substring)
        for substring in substrings:
            substring_counts[substring].append(word)

    #find substrings that are present in at least 4 words
    substrings_by_count = sorted(substring_counts.items(), key=lambda x: len(x[1]), reverse=True)
    for substring, words in substrings_by_count:
        if len(words) >= 4:
            matched_words = words[:4]
            new_guess_set = frozenset(matched_words)
            if new_guess_set not in previous_guesses_sets:
                return [{
                    "category": f"Contains '{substring}'",
                    "matched_words": matched_words
                }]
    return None

@app.route('/')
def index():
    return "Connections Game API. Use the /find_category endpoint."

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Connections Game API with two specified JSON datasets.")
    parser.add_argument('dataset1', type=str, help="Path to the first JSON dataset file")
    parser.add_argument('dataset2', type=str, help="Path to the second JSON dataset file")
    args = parser.parse_args() # easiest way to parse 2 json datasets yuhh

    # Load data from the specified dataset files
    load_data(args.dataset1, args.dataset2)

    # Start the Flask app
    app.run(host='0.0.0.0', port=5001, debug=True)
