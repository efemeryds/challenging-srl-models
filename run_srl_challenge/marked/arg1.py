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


def get_argument(pred, arg_target='B-ARG1'):
    # assume one predicate:
    predicate_arguments = pred['verbs'][0]
    words = pred['words']
    tags = predicate_arguments['tags']
    arg_list = []
    for t, w in zip(tags, words):
        arg = t
        if arg == arg_target:
            arg_list.append(w)
    return arg_list


def pred1_get_argument(pred, arg_target='B-ARG1'):
    # assume one predicate:
    predicate_arguments = pred['verbs'][1]
    words = pred['words']
    tags = predicate_arguments['tags']
    arg_list = []
    for t, w in zip(tags, words):
        arg = t
        if arg == arg_target:
            arg_list.append(w)
    return arg_list


def format_srl(x, pred, conf, label=None, meta=None):
    predicate_structure = pred['verbs'][0]['description']
    return predicate_structure


def pred1_format_srl(x, pred, conf, label=None, meta=None):
    predicate_structure = pred['verbs'][1]['description']
    return predicate_structure


def found_arguments(x, pred, conf, label=None, meta=None):
    # thing = meta['verb'].split()
    thing = ['me']
    predicted_label = get_argument(pred, arg_target='B-ARG1')
    found = False
    if predicted_label:
        if predicted_label[0] in ' '.join(thing):
            found = True
    return found


def pred1_found_arguments(x, pred, conf, label=None, meta=None):
    # thing = meta['verb'].split()
    thing = ['me']
    predicted_label = pred1_get_argument(pred, arg_target='B-ARG1')
    found = False
    if predicted_label:
        if predicted_label[0] in ' '.join(thing):
            found = True
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


def normal(sentence, vocab, model_name):
    expect_arg1 = Expect.single(found_arguments)

    editor = Editor()
    t = editor.template(sentence, meta=True,
                        verb=vocab)

    if model_name == "bert":
        predict_and_conf = PredictorWrapper.wrap_predict(bert_prediction)
    else:
        predict_and_conf = PredictorWrapper.wrap_predict(basic_model_prediction)
    test = MFT(**t, name='detect', expect=expect_arg1)
    test.run(predict_and_conf)
    test.summary(format_example_fn=format_srl)
    return test


def special(sentence, vocab, model_name):
    expect_arg1 = Expect.single(pred1_found_arguments)
    editor = Editor()
    t = editor.template(sentence, meta=True,
                        verb=vocab)
    if model_name == "bert":
        predict_and_conf = PredictorWrapper.wrap_predict(bert_prediction)
    else:
        predict_and_conf = PredictorWrapper.wrap_predict(basic_model_prediction)
    test = MFT(**t, name='detect', expect=expect_arg1)
    test.run(predict_and_conf)
    test.summary(format_example_fn=pred1_format_srl)
    return test


def run_test(sentence, vocab, model_name, gold='B-ARG1', if_special='no'):
    if if_special == "yes":
        test = special(sentence, vocab, model_name)

    else:
        test = normal(sentence, vocab, model_name)

    # create final df
    evaluation_df = pd.DataFrame({'vocab': vocab})
    evaluation_df['sentence'] = sentence

    if if_special == 'yes':
        evaluation_df['if_marked'] = 1
    else:
        evaluation_df['if_marked'] = 0

    evaluation_df['gold'] = gold
    #evaluation_df['expected'] = list(np.concatenate(list(test.results['expect_results'])).ravel())
    evaluation_df['predicted'] = list(test.results['passed'])
    evaluation_df['eval'] = np.where(evaluation_df['predicted'] == True, 1, 0)
    evaluation_df['model_name'] = model_name
    evaluation_df = evaluation_df[['sentence', 'vocab', 'model_name', 'gold', 'eval', 'if_marked']]
    return evaluation_df


# def merge_models_outputs(model1, model2, model3, model4, model5, model6, model7, model8,
#                          model9, model10, model11, model12, output_file):
#     final_data = pd.concat([model1, model2, model3, model4, model5, model6, model7, model8,
#                             model9, model10, model11, model12], ignore_index=True)
#     final_data.to_csv(f"../../evaluation/{output_file}.csv", index=False)
#     print('DONE')

def merge_models_outputs(model1, model2, model3, model4, model5, model6, model7, model8, output_file):
    final_data = pd.concat([model1, model2, model3, model4, model5, model6, model7, model8], ignore_index=True)
    final_data.to_csv(f"../../outcome/{output_file}.csv", index=False)
    print('DONE')


if __name__ == "__main__":
    # load model and inspect output
    bert_model = 'structured-prediction-srl-bert'
    basic_model = 'structured-prediction-srl'

    sentence1 = "She {verb} me."
    sentence2 = "It was her that {verb} me."
    #sentence3 = "Who she {verb} was me."

    with open('../../challenge_tests/vocab/processed_lists.json') as json_file:
        data = json.load(json_file)

    frequent_nouns = data['low_freq_verbs']
    non_frequent_nouns = data['high_freq_verbs']

    basic_f_normal1 = run_test(sentence1, frequent_nouns, 'basic')
    basic_f_normal1['if_frequent'] = 1
    bert_f_normal1 = run_test(sentence1, frequent_nouns, 'bert')
    bert_f_normal1['if_frequent'] = 1

    basic_nf_normal1 = run_test(sentence1, non_frequent_nouns, 'basic')
    basic_nf_normal1['if_frequent'] = 0
    bert_nf_normal1 = run_test(sentence1, non_frequent_nouns, 'bert')
    bert_nf_normal1['if_frequent'] = 0

    basic_f_normal2 = run_test(sentence2, frequent_nouns, 'basic', if_special='yes')
    basic_f_normal2['if_frequent'] = 1
    bert_f_normal2 = run_test(sentence2, frequent_nouns, 'bert', if_special='yes')
    bert_f_normal2['if_frequent'] = 1

    basic_nf_normal2 = run_test(sentence2, non_frequent_nouns, 'basic', if_special='yes')
    basic_nf_normal2['if_frequent'] = 0
    bert_nf_normal2 = run_test(sentence2, non_frequent_nouns, 'bert', if_special='yes')
    bert_nf_normal2['if_frequent'] = 0

    # basic_f_neg3 = run_test(sentence3, frequent_nouns, 'basic')
    # basic_f_neg3['if_frequent'] = 1
    # bert_f_neg3 = run_test(sentence3, frequent_nouns, 'bert')
    # bert_f_neg3['if_frequent'] = 1
    #
    # basic_nf_neg3 = run_test(sentence3, non_frequent_nouns, 'basic')
    # basic_nf_neg3['if_frequent'] = 0
    # bert_ng_neg3 = run_test(sentence3, non_frequent_nouns, 'bert')
    # bert_ng_neg3['if_frequent'] = 0

    #merge_models_outputs(basic_f_normal1, bert_f_normal1, basic_nf_normal1, bert_nf_normal1,
    #                     basic_f_normal2, bert_f_normal2, basic_nf_normal2, bert_nf_normal2,
    #                     basic_f_neg3, bert_f_neg3, basic_nf_neg3, bert_ng_neg3, "marked")

    merge_models_outputs(basic_f_normal1, bert_f_normal1, basic_nf_normal1, bert_nf_normal1,
                        basic_f_normal2, bert_f_normal2, basic_nf_normal2, bert_nf_normal2, "marked")
