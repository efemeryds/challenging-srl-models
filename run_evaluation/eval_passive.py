""" Evaluation of the outcome """

import pandas as pd


def count_failure(input_data):
    count = (((input_data['eval'] == 0).sum() + 0.0001) / len(input_data)) * 100
    count = round(count, 2)
    return count


def get_evaluation(input_data):
    tmp_list = []

    basic = input_data[input_data['model_name'] == 'basic']
    bert = input_data[input_data['model_name'] == 'bert']

    passive_basic_f = count_failure(basic[(basic['if_passive'] == 1) & (basic['if_frequent'] == 1)])
    passive_basic_nf = count_failure(basic[(basic['if_passive'] == 1) & (basic['if_frequent'] == 0)])

    not_passive_basic_f = count_failure(basic[(basic['if_passive'] == 0) & (basic['if_frequent'] == 1)])
    not_passive_basic_nf = count_failure(basic[(basic['if_passive'] == 0) & (basic['if_frequent'] == 0)])

    passive_bert_f = count_failure(bert[(bert['if_passive'] == 1) & (bert['if_frequent'] == 1)])
    passive_bert_nf = count_failure(bert[(bert['if_passive'] == 1) & (bert['if_frequent'] == 0)])

    not_passive_bert_f = count_failure(bert[(bert['if_passive'] == 0) & (bert['if_frequent'] == 1)])
    not_passive_bert_nf = count_failure(bert[(bert['if_passive'] == 0) & (bert['if_frequent'] == 0)])

    tmp_list.append({"model": "bert", "test_type": "passive_freq", "failure_rate (%)": passive_bert_f})
    tmp_list.append({"model": "bert", "test_type": "passive_non_freq", "failure_rate (%)": passive_bert_nf})
    tmp_list.append({"model": "bert", "test_type": "not_passive_freq", "failure_rate (%)": not_passive_bert_f})
    tmp_list.append({"model": "bert", "test_type": "not_passive_non_freq", "failure_rate (%)": not_passive_bert_nf})

    tmp_list.append({"model": "basic", "test_type": "passive_freq", "failure_rate (%)": passive_basic_f})
    tmp_list.append({"model": "basic", "test_type": "passive_non_freq", "failure_rate (%)": passive_basic_nf})
    tmp_list.append({"model": "basic", "test_type": "not_passive_freq", "failure_rate (%)": not_passive_basic_f})
    tmp_list.append({"model": "basic", "test_type": "not_passive_non_freq", "failure_rate (%)": not_passive_basic_nf})
    return tmp_list


results_arg1 = pd.read_csv("../outcome/passive_b_arg1.csv")
final_arg1 = pd.DataFrame(get_evaluation(results_arg1))
final_arg1.to_csv("../evaluation/passive_b_arg1.csv", index=False)

print("DONE")
