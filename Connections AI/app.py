# This is our finished project
# We used an API to mask our openai and huggingface api keys as well as because the models were simply too large to submit (They run incredibly fast though)
# It takes about 15 seconds to start up and then finishes lightning fast.
from flask import Flask, request

# Please do NOT modify this file
# Modifying this file may cause your submission to not be graded

app = Flask(__name__)
@app.post("/")
def challengeSetup():
	req_data = request.get_json()
	words = req_data['words']
	strikes = req_data['strikes']
	isOneAway = req_data['isOneAway']
	correctGroups = req_data['correctGroups']
	previousGuesses = req_data['previousGuesses']
	error = req_data['error']

	guess, endTurn = model(words, strikes, isOneAway, correctGroups, previousGuesses, error)

	return {"guess": guess, "endTurn": endTurn}

if __name__ == '__main__':
    app.run(debug=True, port=5000)




import json
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
def model(words: list, strikes, isOneAway, correctGroups, previousGuesses, error):
    """
    _______________________________________________________
    Parameters:
    words - 1D Array with 16 shuffled words
    strikes - Integer with number of strikes
    isOneAway - Boolean if your previous guess is one word away from the correct answer
    correctGroups - 2D Array with groups previously guessed correctly
    previousGuesses - 2D Array with previous guesses
    error - String with error message (0 if no error)

    Returns:
    guess - 1D Array with 4 words
    endTurn - Boolean if you want to end the puzzle
    _______________________________________________________
    """
    words = eval(words)
   # print("Initial words:", words)

        # Remove words that are in correctGroups
    words_to_remove = set()
    for group in correctGroups:
            for word in group:
                words_to_remove.add(word)
    words = [word for word in words if word not in words_to_remove]
    if(len(words)==4): 
        print("returned via elimination")
        return words,False
   # print("Words after removal:", words)
   # print(previousGuesses)
    returnlist = testcall_all_matches(words,isOneAway,previousGuesses)
    if(len(returnlist)==4): 
        print('returnlisted')
        return returnlist, False
    # Remove words in correctGroups from words
    correct_words = set()
    for group in correctGroups:
        correct_words.update(group)
    words = [word for word in words if word not in correct_words]

    if len(words) == 0:
        # All words have been guessed
        print('LENWORDS0')
        return [], True

    # Load a pre-trained model for word embeddings
    model_embed = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute embeddings for each word
    embeddings = model_embed.encode(words)

    # Determine the number of clusters
    total_words = len(words)
    num_clusters = total_words // 4
    remainder = total_words % 4
    if remainder != 0:
        num_clusters += 1  # Include the last cluster with fewer words

    if total_words >= 4:
        # Use Agglomerative Clustering
        clustering_model = AgglomerativeClustering(n_clusters=num_clusters, metric='cosine', linkage='average')
        clustering_model.fit(embeddings)
        labels = clustering_model.labels_

        # Build clusters
        cluster_dict = {}
        for idx, label in enumerate(labels):
            cluster_dict.setdefault(label, []).append(words[idx])

        # Flatten clusters into a list and adjust to have clusters of size 4
        all_words = []
        for cluster_words in cluster_dict.values():
            all_words.extend(cluster_words)

        clusters_of_size_4 = [all_words[i:i + 4] for i in range(0, len(all_words), 4)]
    else:
        # Less than 4 words left
        clusters_of_size_4 = [words]

    # Preprocess previousGuesses into sets
    previousGuesses_sets = [set(guess) for guess in previousGuesses]

    # Get last guess
    if previousGuesses:
        last_guess = previousGuesses[-1]
        # Remove words not in current 'words' (they may have been guessed correctly)
        last_guess = [word for word in last_guess if word in words]
        if len(last_guess) == 0:
            last_guess = None
    else:
        last_guess = None

    selected_cluster = None
    if last_guess:
        last_guess_set = set(last_guess)
    else:
        last_guess_set = None

    if isOneAway and last_guess_set:
        # Focus on clusters similar to the last guess
        # Compute the embedding for the last guess by averaging embeddings
        last_guess_embeddings = model_embed.encode(last_guess)
        last_guess_embedding = np.mean(last_guess_embeddings, axis=0)

        # Initialize variables to track the best cluster
        best_cluster = None
        highest_similarity = -1

        for cluster in clusters_of_size_4:
            if len(cluster) != 4:
                continue  # Ensure cluster size is 4

            cluster_set = set(cluster)

            # Check if cluster has been guessed before
            if any(cluster_set == prev_guess_set for prev_guess_set in previousGuesses_sets):
                continue

            # Compute the average similarity of the cluster to the last guess
            cluster_embeddings = model_embed.encode(cluster)
            avg_similarity = np.mean([
                cosine_similarity(last_guess_embedding, cluster_emb)
                for cluster_emb in cluster_embeddings
            ])

            # Update the best cluster based on similarity
            if avg_similarity > highest_similarity:
                best_cluster = cluster
                highest_similarity = avg_similarity

        if best_cluster:
            selected_cluster = best_cluster

    if not selected_cluster:
        # Fallback to original strategy when not one away or no suitable cluster found
        for cluster in clusters_of_size_4:
            if len(cluster) != 4:
                continue  # Ensure cluster size is 4

            cluster_set = set(cluster)

            # Check if cluster has been guessed before
            if any(cluster_set == prev_guess_set for prev_guess_set in previousGuesses_sets):
                continue

            if isOneAway and last_guess_set:
                # Calculate symmetric difference
                diff = len(cluster_set.symmetric_difference(last_guess_set))
                if diff != 2:
                    continue  # Skip clusters that are not one away
            # For non-one-away cases or if conditions are met
            selected_cluster = cluster
            break

    if selected_cluster is None:
        # Cannot find a cluster meeting criteria, select any cluster of size 4 not guessed before
        for cluster in clusters_of_size_4:
            if len(cluster) != 4:
                continue  # Ensure cluster size is 4
            cluster_set = set(cluster)
            if any(cluster_set == prev_guess_set for prev_guess_set in previousGuesses_sets):
                continue
            selected_cluster = cluster
            break

    if selected_cluster is None:
        # No clusters of size 4 available, end the turn or return remaining words
        if len(words) <= 4:
            selected_cluster = words
        else:
            print("LENWORDSMORETHAN4    ")
            print(words)
            return [], True  # No valid guesses left

    # Return the selected cluster
    return selected_cluster, False


def testcall_all_matches(word_list,isOneAway,previousGuesses): 
    securl = 'http://10.247.181.137:5001/find_category' # VERY LARGE LLM hosted here with massive context
    payload = {'words': word_list,'isOneAway':isOneAway,'previousGuesses':previousGuesses}
    all_matched_words = []
    try:
        response = requests.post(securl, json=payload, timeout=10)
        response.raise_for_status()
        response_data = response.json()
        if 'matches' in response_data and isinstance(response_data['matches'], list):
            for match in response_data['matches']:
                if 'matched_words' in match and isinstance(match['matched_words'], list):
                    all_matched_words.extend(match['matched_words'][:4])
        return all_matched_words[:4]  # Return only the first four matched words overall
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return [],True

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
      