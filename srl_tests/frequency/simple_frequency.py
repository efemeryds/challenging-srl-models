from allennlp_models.pretrained import load_predictor
import nltk
import spacy

# nltk.download('omw-1.4')
spacy.load('en_core_web_sm')
import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.pred_wrapper import PredictorWrapper

# load model and inspect output
bert_model = 'structured-prediction-srl-bert'
basic_model = 'structured-prediction-srl'

srl_predictor = load_predictor(basic_model)


""" SIMPLE TEST
Check whether the model is able to correctly assign ArgM-LOC depending on the frequency of the words """

output = srl_predictor.predict("It happened .")
print(output)


""" PREPARE INPUT WORDS -> """

def get_arg(pred, arg_target='ARG1'):
    # we assume one predicate:
    predicate_arguments = pred['verbs'][0]
    words = pred['words']
    tags = predicate_arguments['tags']

    arg_list = []
    for t, w in zip(tags, words):
        arg = t
        if '-' in t:
            arg = t.split('-')[1]
        if arg == arg_target:
            arg_list.append(w)
    arg_set = set(arg_list)
    return arg_set


# Helper function to display failures

def format_srl(x, pred, conf, label=None, meta=None):
    predicate_structure = pred['verbs'][0]['description']
    return predicate_structure


def found_arg1_people(x, pred, conf, label=None, meta=None):
    # people should be recognized as arg1

    people = meta['location']
    arg_1 = get_arg(pred, arg_target='ArgM-LOC')

    if arg_1 == people:
        pass_ = True
    else:
        pass_ = False
    return pass_


expect_argm_loc = Expect.single(found_arg1_people)

editor = Editor()

vocab = ["far far away", "in the Wonderland", "next to my home", "on the street", "hidden in the forest"]

# create examples
t = editor.template("When I was little someone told me about a place {location}.", meta=True, location=vocab)

print(type(t))

for k, v in t.items():
    print(k, v)


def predict_srl(data):
    pred = []
    for d in data:
        pred.append(srl_predictor.predict(d))
    return pred


predict_and_conf = PredictorWrapper.wrap_predict(predict_srl)
# initialize a rest object
test = MFT(**t, name='detect_ArgM-LOC_position', expect=expect_argm_loc)
test.run(predict_and_conf)
test.summary(format_example_fn=format_srl)

print('DONE')






