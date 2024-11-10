import random
import numpy as np
import requests
import json

def evalFunction():
	# Load puzzles
	puzzles = load_puzzles()
	totalPoints = 0

	z = 1
	invalidGuesses = 0

	for puzzle in puzzles:
		shuffledPuzzle = shufflePuzzles(puzzle)

		# Initialize all variables
		strikes = 0
		correctGroups = []
		previousGuesses = []
		error = 0
		isOneAway = False

		while strikes < 4 and len(correctGroups) < 4 and invalidGuesses < 7:
			data = {
				"words": shuffledPuzzle,
				"strikes": strikes,
				"isOneAway": isOneAway,
				"correctGroups": correctGroups,
				"previousGuesses": previousGuesses,
				"error": error
			}
			headers = {'Content-Type': 'application/json'}

			# participantGuess, endTurn = participantModel(shuffledPuzzle, strikes, isOneAway, correctGroups, previousGuesses, error)
			r = requests.post("http://127.0.0.1:5000", data=json.dumps(data), headers={'Content-Type': 'application/json'})
			participantGuess = r.json()['guess']
			print("Participant guess: ", participantGuess)
			endTurn = r.json()['endTurn']
			sortedGuess = np.sort(participantGuess)
			if any(np.array_equal(sortedGuess, x) for x in previousGuesses):
				error = "You have already guessed this combination."
				invalidGuesses += 1
				continue
			else:
				error = 0
				previousGuesses.append(participantGuess)

			if len(participantGuess) != 4:
				error = "Please enter 4 words."
				invalidGuesses += 1
				continue

			if endTurn:
				break

			correctlyGuessed = False
			for group in puzzle:
				sortedPuzzle = np.sort(group)
				if np.array_equal(sortedPuzzle, sortedGuess):
					correctlyGuessed = True
					correctGroups.append(group)
					break
				else:
					set1, set2 = set(group), set(participantGuess)
					if (len(set1.symmetric_difference(set2)) == 2):
						isOneAway = True
						print("One away")
						break
					else:
						isOneAway = False

			if not correctlyGuessed:
				strikes += 1


		# Calculate points
		points = 0
		groupMult = 1
		strikeMult = 1

		for i in range(len(correctGroups)):
			match i + 1:
				case 1:
					groupMult = 1
				case 2:
					groupMult = 2
				case 3:
					groupMult = 3
				case 4:
					groupMult = 3

			match strikes:
				case 0:
					strikeMult = 1
				case 1:
					strikeMult = 0.9
				case 2:
					strikeMult = 0.75
				case 3:
					strikeMult = 0.5
				case 4:
					strikeMult = 0.25

			points += groupMult * strikeMult
			print("Points scored by model on puzzle " +  str(z) + ", group ", i+1, ": ", groupMult * strikeMult)
		totalPoints += points
		z += 1

	# Store total points
	print("Total points scored by model: ", totalPoints)

def load_puzzles():
    with open('sample_data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Create a 3D array (X puzzles, 4 rows, 4 words)
    puzzles_3d = []

    for puzzle in data:
        # Extract only the words part for each puzzle
        puzzle_words = [entry["words"] for entry in puzzle]
        puzzles_3d.append(puzzle_words)

    return puzzles_3d

def shufflePuzzles(puzzle):
	# Return puzzle shuffled as a 1d array
	flattenedPuzzle = np.array(puzzle).reshape(-1)
	np.random.shuffle(flattenedPuzzle)
	return np.array2string(flattenedPuzzle, separator=', ')

evalFunction()
