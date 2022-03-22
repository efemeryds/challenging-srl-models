Reading guide for Ribeiro et al 2020

Try to answer the following questions when reading Ribeiro et al. 2020:
    a) Motivation and approach:

i) What is meant by bias in training data? What does it have to do with the 
fact that we tend to overestimate real-world performance of systems?

ii) What is meant by behavioral testing?

iii) Which three NLP tasks are used to showcase checklist? Provide a short
explanation of each task

iv) How is the checklist approach being evaluated?

b) Checklist

i) What does the checklist matrix look like? What do the rows and columns
represent?

ii) How should we interpret this matrix? Is it necessary/possible to fill all cells
with specific tests?

iii) Test instance generation: How are instances generated? What role can
masked language models play?

c) Task-specific checklists

i) Consider the matrix represented in Tables 1, 2, and 3. Go through the
different tests and try to understand the test types and what they tes



Notes:
CheckList includes a matrix of general linguistic capabilities and test types that
facilitate comprehensive test ideation, as well as a software tool to generate a large and
diverse number of test cases quickly.

One of the primary goals of training NLP models is generalization.
Since testing “in the wild” is expensive and does not allow for fast iterations,
the standard paradigm for evaluation is using train-validation-test splits
to estimate the accuracy of the model, including the use of leader boards to
track progress on a task (Rajpurkar et al., 2016). While performance on held-out
data is a useful indicator, held-out datasets are often not comprehensive,
and contain the same biases as the training data.


Tests are structured as a conceptual matrix with capabilities as rows and test
types as columns



