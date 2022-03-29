from checklist.editor import Editor
from checklist.test_types import MFT
from checklist.expect import Expect
from checklist.pred_wrapper import PredictorWrapper
from allennlp_models.pretrained import load_predictor
import logging
import json
# nltk.download('omw-1.4')
import spacy
import pandas as pd
import numpy as np

spacy.load('en_core_web_sm')

logging.getLogger('allennlp.common.params').disabled = True
logging.getLogger('allennlp.nn.initializers').disabled = True
logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').disabled = True
logging.getLogger('allennlp.common.plugins').disabled = True
logging.getLogger('allennlp.models.archival').disabled = True


def get_argument(pred, arg_target='I-ARG2'):
    # assume one predicate:
    predicate_arguments = pred['verbs'][0]
    words = pred['words']
    tags = predicate_arguments['tags']

    arg_list = []
    for t, w in zip(tags, words):
        arg = t
        # if len(t) > 2:
        #    if t[1] == '-':
        #        arg = t[2:]
        if arg == arg_target:
            arg_list.append(w)
    return arg_list


def format_srl(x, pred, conf, label=None, meta=None):
    predicate_structure = pred['verbs'][0]['description']
    return predicate_structure


def found_arguments(x, pred, conf, label=None, meta=None):
    thing = meta['vocab'].split()
    predicted_label = get_argument(pred, arg_target='I-ARG2')
    if predicted_label == thing:
        found = True
    else:
        found = False
    return found


def basic_model_prediction(data):
    srl_predictor = load_predictor('structured-prediction-srl')
    predicate_list = []
    for d in data:
        predicate_list.append(srl_predictor.predict(d))
    return predicate_list


def bert_prediction(data):
    srl_predictor = load_predictor('structured-prediction-srl-bert')
    predicate_list = []
    for d in data:
        predicate_list.append(srl_predictor.predict(d))
    return predicate_list


def run_test(sentence, vocab, model_name, gold='I-ARG2'):
    expect_arg1 = Expect.single(found_arguments)

    editor = Editor()
    t = editor.template(sentence, meta=True,
                        vocab=vocab)

    if model_name == "bert":
        predict_and_conf = PredictorWrapper.wrap_predict(bert_prediction)
    else:
        predict_and_conf = PredictorWrapper.wrap_predict(basic_model_prediction)
    test = MFT(**t, name='detect', expect=expect_arg1)
    test.run(predict_and_conf)
    test.summary(format_example_fn=format_srl)

    # create final df
    evaluation_df = pd.DataFrame({'vocab': vocab})
    evaluation_df['sentence'] = sentence
    evaluation_df['gold'] = gold
    evaluation_df['expected'] = list(np.concatenate(list(test.results['expect_results'])).ravel())
    evaluation_df['predicted'] = list(test.results['passed'])
    evaluation_df['eval'] = np.where(evaluation_df['expected'] == evaluation_df['predicted'], 1, 0)
    evaluation_df['model_name'] = model_name
    evaluation_df = evaluation_df[['sentence', 'vocab', 'model_name', 'gold', 'eval']]
    return evaluation_df


def merge_models_outputs(model1, model2, model3, model4, model5, model6, model7, model8, output_file):
    final_data = pd.concat([model1, model2, model3, model4, model5, model6, model7, model8], ignore_index=True)
    final_data.to_csv(f"../../evaluation/{output_file}.csv")
    print('DONE')


if __name__ == "__main__":
    # load model and inspect output
    bert_model = 'structured-prediction-srl-bert'
    basic_model = 'structured-prediction-srl'

    input_vocab = ["far away", "in the Wonderland", "next to my home", "on the street", "in the forest",
                   "in the hospital"]

    input_sentence = "She hurt him with {vocab}."
    input_negation_sentence = "She didn't hurt him with {vocab}."

    with open('../../challenge_tests/vocab/processed_lists.json') as json_file:
        data = json.load(json_file)

    frequent_nouns = data['low_freq_objects']
    non_frequent_nouns = data['high_freq_objects']

    basic_f_normal = run_test(input_sentence, frequent_nouns, 'basic')
    basic_f_normal['if_frequent'] = 1
    bert_f_normal = run_test(input_sentence, frequent_nouns, 'bert')
    bert_f_normal['if_frequent'] = 1

    basic_nf_normal = run_test(input_sentence, non_frequent_nouns, 'basic')
    basic_nf_normal['if_frequent'] = 0
    bert_nf_normal = run_test(input_sentence, non_frequent_nouns, 'bert')
    bert_nf_normal['if_frequent'] = 0

    basic_f_neg = run_test(input_negation_sentence, frequent_nouns, 'basic')
    basic_f_neg['if_frequent'] = 1
    bert_f_neg = run_test(input_negation_sentence, frequent_nouns, 'bert')
    bert_f_neg['if_frequent'] = 1

    basic_nf_neg = run_test(input_negation_sentence, non_frequent_nouns, 'basic')
    basic_nf_neg['if_frequent'] = 0
    bert_ng_neg = run_test(input_negation_sentence, non_frequent_nouns, 'bert')
    bert_ng_neg['if_frequent'] = 0

    merge_models_outputs(basic_f_normal, bert_f_normal, basic_nf_normal, bert_nf_normal,
                         basic_f_neg, bert_f_neg, basic_nf_neg, bert_ng_neg, "negation_arg2")

