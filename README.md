# Challenging Semantic Role Labelling (SRL) models
The aim of this project is to create a challenge dataset for the semantic role labelling task.

## Semantic Role Labelling
- In a nutshell, semantic role labelling is a task of assigning roles to spans in given sentences. 
Semantic roles are crucial in obtaining basic semantic representation of a given text, like "who
does what to whom when, where, etc.". 
- One can use SRL system to classify arguments of predicates. In order to do that, one needs
certain labels. The motivation for using semantic roles is to move from sentences towards propositions.
The same information can be encoded using different syntactic variations but if we stick to proposition
representation, we should be able to treat all of them as expressing same phenomenon. 
- Formally the task of SRL can be defined as identifying the predicate argument structures.
- Properly working SRL system should be able to receive:
  - input: as a text
  
  And return:
  - output: as a sequence of labeled arguments
- The system works in a given sequence given it gets the information on the predicate:
  1. Perform argument identification - different ways of achieving that, can be hand-written rules, or more sophisticated algorithmic methods
  2. Perform role labeling - among other possibilities, one can use SVM or logistic regression to get most probable label
  3. Perform inference - among other options, one can introduce certain linguistic and structural constraints 
- Features used to train SRL systems are based on different syntactic
categories or dependency labels. The examples of chosen features are listed below:

  - named entities 
  - word embeddings 
  - PoS-tag of the previous token 
  - PoS-tag of the next token 
  - tokens
  - lemmas
  - morphological features
  - head words


## List of capabilities that can be investigated
- Syntactic variation
  - Statement vs question
  - Active vs passive
- Predicates, argument patterns and predicate senses
- Lexicalizations of arguments
  - frequent vs infrequent words
  - Proper names
- Negation


## Models that are being tested
The models come from AllenNLP project https://github.com/allenai/allennlp-models


### First model: structured-prediction-srl-bert
A BERT based model (Shi et al, 2019).



### Second model: structured-prediction-srl 
A reimplementation of a deep BiLSTM sequence prediction model (Stanovsky et al., 2018).


## Example tests

### Test 1
Low and high frequency words - choice based on the frequencies from the corpus used in a
given model. Test the ability of correctly labelling the predicates and arguments 
on simple sentences. 


a) motivation


b) hypothesis

c) model output


### Test 2
Low and high frequency words with negation

a) motivation


b) hypothesis

c) model output


### Test 3
Low and high frequency words with ... 



### Test 4
Low and high frequency words with ... 




### Test 5
Low and high frequency words with ... 


## How to run the project


## Conclusions


## Sources:
- Slides from the NLP Technology course at Vrije Universiteit Amsterdam
- https://web.stanford.edu/~jurafsky/slp3/19.pdf
- https://docs.allennlp.org/models/main/