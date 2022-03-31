""" Prints all the results """

import pandas as pd

print("Lexicalizations baseline")

print("ARG1")
data = pd.read_csv("evaluation/frequency_arg1.csv")
print(data)
print("------------------------------------------")

print("ARGM-LOC")
data = pd.read_csv("evaluation/frequency_argm_loc.csv")
print(data)
print("------------------------------------------")

print("I-ARG2")
data = pd.read_csv("evaluation/instrument_arg2.csv")
print(data)
print("------------------------------------------")

print("Marked vs unmarked")
data = pd.read_csv("evaluation/marked.csv")
print(data)
print("------------------------------------------")

print("Active vs passive")
data = pd.read_csv("evaluation/passive_b_arg1.csv")
print(data)
print("------------------------------------------")

print("Statement vs question")
data = pd.read_csv("evaluation/question.csv")
print(data)
print("------------------------------------------")

print("Negation")
data = pd.read_csv("evaluation/negation_arg2.csv")
print(data)
print("------------------------------------------")

print("DONE")


