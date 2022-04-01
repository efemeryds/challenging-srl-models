import json
import pandas as pd

# read input words

with open('../challenge_tests/vocab/processed_lists.json') as json_file:
    data = json.load(json_file)

low_freq_places = data['low_freq_places']
high_freq_places = data['high_freq_places']
low_freq_objects = data['low_freq_objects']
high_freq_objects = data['high_freq_objects']
low_freq_verbs = data['low_freq_verbs']
high_freq_verbs = data['high_freq_verbs']

# loop over the sentences and save them to df

print("Lexicalizations baseline")

# ARG1
gold = "ARG1"

sent = []
words = []
for word in low_freq_objects:
    example = f"Someone stole {word} from his garage."
    sent.append(example)
    words.append(word)

for word in high_freq_objects:
    example = f"Someone stole {word} from his garage."
    sent.append(example)
    words.append(word)

print(gold)
print(sent)

baseline_arg1 = pd.DataFrame({"sentence": sent, "target": words, "gold": gold})
baseline_arg1.to_csv("../challenge_tests/sentences/baseline_arg1.csv", index=False)

# I-ARGM-LOC
gold = "I-ARGM-LOC"

sent = []
words = []
for word in low_freq_places:
    example = f"Someone told me a secret, it happened next to the {word}."
    sent.append(example)
    words.append(word)

for word in high_freq_places:
    example = f"Someone told me a secret, it happened next to the {word}."
    sent.append(example)
    words.append(word)

print(gold)
print(sent)

baseline_i_argm_loc = pd.DataFrame({"sentence": sent, "target": words, "gold": gold})
baseline_i_argm_loc.to_csv("../challenge_tests/sentences/baseline_i_argm_loc.csv", index=False)


# I-ARG2
gold = "I-ARG2"

sent = []
words = []
for word in low_freq_objects:
    example = f"She hurt him with the {word}."
    sent.append(example)
    words.append(word)

for word in high_freq_objects:
    example = f"She hurt him with the {word}."
    sent.append(example)
    words.append(word)

print(gold)
print(sent)

baseline_i_arg2 = pd.DataFrame({"sentence": sent, "target": words, "gold": gold})
baseline_i_arg2.to_csv("../challenge_tests/sentences/baseline_i_arg2.csv", index=False)


# Syntactic variation

# Marked B-ARG1
gold = "B-ARG1"

sent = []
words = []
for word in low_freq_verbs:
    example = f"She {word} me."
    sent.append(example)
    words.append("me")

for word in high_freq_verbs:
    example = f"She {word} me."
    sent.append(example)
    words.append("me")

for word in low_freq_verbs:
    example = f"It was her that {word} me."
    sent.append(example)
    words.append("me")

for word in high_freq_verbs:
    example = f"It was her that {word} me."
    sent.append(example)
    words.append("me")

print(gold)
print(sent)

marked_b_arg1 = pd.DataFrame({"sentence": sent, "target": words, "gold": gold})
marked_b_arg1.to_csv("../challenge_tests/sentences/marked_b_arg1.csv", index=False)


# Passive B-ARG1
gold = "B-ARG1"

sent = []
words = []
for word in low_freq_verbs:
    example = f"She was the one who {word} me."
    sent.append(example)
    words.append("me or I")

for word in high_freq_verbs:
    example = f"She was the one who {word} me."
    sent.append(example)
    words.append("me or I")

for word in low_freq_verbs:
    example = f"I was {word} by her."
    sent.append(example)
    words.append("me or I")

for word in high_freq_verbs:
    example = f"I was {word} by her."
    sent.append(example)
    words.append("me or I")

print(gold)
print(sent)

passive_b_arg1 = pd.DataFrame({"sentence": sent, "target": words, "gold": gold})
passive_b_arg1.to_csv("../challenge_tests/sentences/passive_b_arg1.csv", index=False)


# Question B-ARG1
gold = "B-ARG1"

sent = []
words = []
for word in low_freq_objects:
    example = f"He often thinks about {word}."
    sent.append(example)
    words.append("he or He")

for word in high_freq_objects:
    example = f"He often thinks about {word}."
    sent.append(example)
    words.append("he or He")

for word in low_freq_objects:
    example = f"Does he often think about {word}?"
    sent.append(example)
    words.append("he or He")

for word in high_freq_objects:
    example = f"Does he often think about {word}?"
    sent.append(example)
    words.append("he or He")

print(gold)
print(sent)

question_b_arg1 = pd.DataFrame({"sentence": sent, "target": words, "gold": gold})
question_b_arg1.to_csv("../challenge_tests/sentences/question_b_arg1.csv", index=False)

# Negation I-ARG2
gold = "I-ARG2"

input_sentence = "It was her who hurt him with {word}."
input_negation_sentence = "It was not her who hurt him with {word}."

sent = []
words = []
for word in low_freq_objects:
    example = f"It was her who hurt him with {word}."
    sent.append(example)
    words.append(word)

for word in high_freq_objects:
    example = f"It was her who hurt him with {word}."
    sent.append(example)
    words.append(word)

for word in low_freq_objects:
    example = f"It was not her who hurt him with {word}."
    sent.append(example)
    words.append(word)

for word in high_freq_objects:
    example = f"It was not her who hurt him with {word}."
    sent.append(example)
    words.append(word)


print(gold)
print(sent)

negation_b_arg1 = pd.DataFrame({"sentence": sent, "target": words, "gold": gold})
negation_b_arg1.to_csv("../challenge_tests/sentences/negation_b_arg1.csv", index=False)

print("DONE")
