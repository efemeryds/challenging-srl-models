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
srl_predictor = load_predictor('structured-prediction-srl')
output = srl_predictor.predict("The killer killed the victim with a knife.")

""" RECREATING EXAMLPE WITH NAMES """


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
    results = []
    predicate_structure = pred['verbs'][0]['description']

    return predicate_structure


def found_arg1_people(x, pred, conf, label=None, meta=None):
    # people should be recognized as arg1

    people = set([meta['first_name'], meta['last_name']])
    arg_1 = get_arg(pred, arg_target='ARG1')

    if arg_1 == people:
        pass_ = True
    else:
        pass_ = False
    return pass_


def predict_srl(data):
    pred = []
    for d in data:
        pred.append(srl_predictor.predict(d))
    return pred

expect_arg1 = Expect.single(found_arg1_people)
editor = Editor()

# create examples
t = editor.template("Someone killed {first_name} {last_name} last night.", meta=True, nsamples=10)
# print(type(t))
# for k, v in t.items():
#     print(k, v)

predict_and_conf = PredictorWrapper.wrap_predict(predict_srl)
# initialize a rest object
test = MFT(**t, name='detect_arg1_name_default_position', expect=expect_arg1)
test.run(predict_and_conf)
test.summary(format_example_fn=format_srl)

print('DONE')
