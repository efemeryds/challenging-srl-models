# Challenging Semantic Role Labelling (SRL) models
The aim of this project is to create a challenge dataset for the semantic role labelling task.


## Semantic Role Labelling
- Semantic role labeling (SRL) is a task concerning the classification of the linguistic phenomena.
- The aim of such classification is to gain more information on a given text input. SRL enables one to obtain meaningful inferences by representing a text in a form: who did what to whom, how, with what, when and where.
- To achieve such representation the system identifies predicates together with their arguments that belong to a specified thematic roles.


## List of capabilities
- Syntactic variation
  - Statement vs question
  - Active vs passive
  - Marked vs unmarked
- Lexicalizations of arguments
  - frequent vs infrequent words
  - Proper names
- Negation


## Models that are being tested
- structured-prediction-srl-bert
- structured-prediction-srl 

The models come from AllenNLP project https://github.com/allenai/allennlp-models


## How to use the project
- Execute **run_evaluation_results.py** to see the summarized performance of all of the tests
- Go to **challenge_tests/sentences/** to see the challenge sets per each test
- Go to **run_srl_challenge/** to investigate how the tests were created
- Go to **outcome/** to see the outcome per each test and each specific example
