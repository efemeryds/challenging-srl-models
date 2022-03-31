""" Evaluation of the outcome """

import pandas as pd


def count_failure(input_data):
    count = (((input_data['eval'] == 0).sum() + 0.0001) / len(results_arg1)) * 100
    count = round(count, 2)
    return count


def get_evaluation(input_data):
    tmp_list = []

    basic = input_data[input_data['model_name'] == 'basic']
    bert = input_data[input_data['model_name'] == 'bert']

    neg_basic_f = count_failure(basic[(basic['if_negation'] == 1) & (basic['if_frequent'] == 1)])
    neg_basic_nf = count_failure(basic[(basic['if_negation'] == 1) & (basic['if_frequent'] == 0)])

    not_neg_basic_f = count_failure(basic[(basic['if_negation'] == 0) & (basic['if_frequent'] == 1)])
    not_neg_basic_nf = count_failure(basic[(basic['if_negation'] == 0) & (basic['if_frequent'] == 0)])

    neg_bert_f = count_failure(bert[(bert['if_negation'] == 1) & (bert['if_frequent'] == 1)])
    neg_bert_nf = count_failure(bert[(bert['if_negation'] == 1) & (bert['if_frequent'] == 0)])

    not_neg_bert_f = count_failure(bert[(bert['if_negation'] == 0) & (bert['if_frequent'] == 1)])
    not_neg_bert_nf = count_failure(bert[(bert['if_negation'] == 0) & (bert['if_frequent'] == 0)])

    tmp_list.append({"model": "bert", "test_type": "neg_freq", "failure_rate (%)": neg_bert_f})
    tmp_list.append({"model": "bert", "test_type": "neg_non_freq", "failure_rate (%)": neg_bert_nf})
    tmp_list.append({"model": "bert", "test_type": "not_neg_freq", "failure_rate (%)": not_neg_bert_f})
    tmp_list.append({"model": "bert", "test_type": "not_neg_non_freq", "failure_rate (%)": not_neg_bert_nf})

    tmp_list.append({"model": "basic", "test_type": "neg_freq", "failure_rate (%)": neg_basic_f})
    tmp_list.append({"model": "basic", "test_type": "neg_non_freq", "failure_rate (%)": neg_basic_nf})
    tmp_list.append({"model": "basic", "test_type": "not_neg_freq", "failure_rate (%)": not_neg_basic_f})
    tmp_list.append({"model": "basic", "test_type": "not_neg_non_freq", "failure_rate (%)": not_neg_basic_nf})
    return tmp_list


results_arg1 = pd.read_csv("../outcome/negation_arg2.csv")
final_arg1 = pd.DataFrame(get_evaluation(results_arg1))
final_arg1.to_csv("../evaluation/negation_arg2.csv", index=False)

print("DONE")
