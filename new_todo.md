New plan

1. Prepare same structure everywhere with proper expected arguments (1 sentence per each)
      - fix the exemplary sentence in negation file
      - add low and high frequency word lists to the file
      - modify the output of passive file
      - modify the output of question file
      - modify the output of synonyms file 

2. Test that everything works 
3. Prepare evaluation metrics (check which one that was supposed to be)

4. Update readme! after the test changes











# OLD

Current todo:

- Go through papers in the description of models and describe how they work (look at slides)
- Design a structure for final evaluation
- Design pipeline with input sentences and vocab
- More complex examples, "From the house of ... "
- Look at the errors in https://www.researchgate.net/publication/318740296_Deep_Semantic_Role_Labeling_What_Works_and_What%27s_Next
- check github probbank predicate frames


questions:
- how many different arguments to test per test?
- how many sentence structures? 


- default + robustness (low and high frequency)
- ~3-5 different arguments per sentences

test motivations:
- central capability for SRL (in more details)

feedback
- maybe better to not use checklist, but create own pipeline
- final dataset with all examples to input into models and evaluate,
with column with what is expected etc. per each test
- better justification for the checklist
- for specific test be as specific as possible, "passive" too general,
write about arguments that are being tested


refactor:
- write down all general steps 
  - prepare a list of sentence structures to test (template to generate)
    - 5 per each test
      - sometimes more tests per one sentence (if sentence alternation)
  - prepare vocab OK
  - prepare dataset per each test -> table with sentence, vocab, type,
  what_is_checked, correct_answer

  - pipeline:
  - output saved as df as well with information of scores
    - use failure rate from checklist (binary, is it this argument, yes or no)
    - one per each argument test
    - compare if the low frequency influence results (globally and locally)
  - analyse 


groundtruth -> propbank
write code to get predicate from the xml
steps:
1. clone propbank repo
2. lemmas of predicates
3. etree module
4. load the file
5. probbank notebook
6. write in the report that according to the propbank
7. just use the website

refactor code -> universal to the top



Minimal requirements:
- 



General feedback:
- linguistic phenomena 
- capability -> making phenomena concrete and applying it
- be more specific about the capabilities and how they are related to SRL


SRL has to learn:
- voice recognition
- semantics -> even if subject but more instrument (eg hammer)
- verb classes
- list on slides

SRL shortcuts:
- first noun is always an agent
- object is always a theme
- slides

Ideas:
- not popular instruments/not sentient being in subject position (from the embedding
list I have)
- start with simple example and gradually add complexity and see when it breaks
- group it finally by the word frequency
- garden path sentences


Current Plan:
- make a file per each test group (and argument) that creates df
in the runtime while calculating the score
- add 2-3 meaningful sentences per tests (frequency, passive, question, synonyms)
- for negation use the sentences from previous tests
- prepare the vocab for the tests
- start writing the report
- iterate

General idea:
- start with frequency only to get the baseline - use all the prepared vocab
- then check the rest phenomena
- use own code??



