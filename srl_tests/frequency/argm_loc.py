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
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# load model and inspect output
bert_model = 'structured-prediction-srl-bert'
basic_model = 'structured-prediction-srl'

srl_predictor = load_predictor(basic_model)


def get_argument(pred, arg_target='ARG1'):
    # assume one predicate:
    predicate_arguments = pred['verbs'][2]
    words = pred['words']
    tags = predicate_arguments['tags']

    arg_list = []
    for t, w in zip(tags, words):
        arg = t
        if len(t) > 2:
            if t[1] == '-':
                arg = t[2:]
        if arg == arg_target:
            arg_list.append(w)
    return arg_list


def format_srl(x, pred, conf, label=None, meta=None):
    predicate_structure = pred['verbs'][2]['description']
    return predicate_structure


def found_arguments(x, pred, conf, label=None, meta=None):
    location = meta['location'].split()
    argument_loc = get_argument(pred, arg_target='ARGM-LOC')
    if argument_loc == location:
        found = True
    else:
        found = False
    return found


def predict_srl(data):
    predicate_list = []
    for d in data:
        predicate_list.append(srl_predictor.predict(d))
    return predicate_list


def run_location_test(sentence, vocab):
    expect_argm_loc = Expect.single(found_arguments)
    editor = Editor()
    t = editor.template(sentence, meta=True,
                    location=vocab)
    print(type(t))
    for k, v in t.items():
        print(k, v)
    predict_and_conf = PredictorWrapper.wrap_predict(predict_srl)
    test = MFT(**t, name='detect_ArgM-LOC_position', expect=expect_argm_loc)
    test.run(predict_and_conf)
    test.summary(format_example_fn=format_srl)


if __name__ == "__main__":
    input_vocab = ["far away", "in the Wonderland", "next to my home", "on the street", "in the forest", "in the hospital"]
    input_sentence = "When I was younger someone told me about a magical place, this happened {location}."
    run_location_test(input_sentence, input_vocab)
