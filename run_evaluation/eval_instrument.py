""" Evaluation of the outcome """

import pandas as pd


def count_failure(input_data):
    count = (((input_data['eval'] == 0).sum() + 0.0001) / len(results_arg1)) * 100
    count = round(count, 2)
    return count


def get_model_results(input_data):
    frequent = input_data[input_data['if_frequent'] == 1]
    non_frequent = input_data[input_data['if_frequent'] == 0]
    frequent_count = count_failure(frequent)
    infrequent_count = count_failure(non_frequent)
    general_count = count_failure(input_data)
    return general_count, frequent_count, infrequent_count


def get_evaluation(input_data):
    tmp_list = []

    basic = input_data[input_data['model_name'] == 'basic']
    bert = input_data[input_data['model_name'] == 'bert']
    b_general, b_freq, b_nonfreq = get_model_results(basic)
    bert_general, bert_freq, bert_nonfreq = get_model_results(bert)

    tmp_list.append({"model": "bert", "test_type": "general", "failure_rate (%)": b_general})
    tmp_list.append({"model": "bert", "test_type": "frequent", "failure_rate (%)": b_freq})
    tmp_list.append({"model": "bert", "test_type": "non_frequent", "failure_rate (%)": b_nonfreq})

    tmp_list.append({"model": "basic", "test_type": "general", "failure_rate (%)": bert_general})
    tmp_list.append({"model": "basic", "test_type": "frequent", "failure_rate (%)": bert_freq})
    tmp_list.append({"model": "basic", "test_type": "non_frequent", "failure_rate (%)": bert_nonfreq})
    return tmp_list


results_arg1 = pd.read_csv("../outcome/instrument_arg2.csv")
final_arg1 = pd.DataFrame(get_evaluation(results_arg1))
final_arg1.to_csv("../evaluation/instrument_arg2.csv", index=False)

print("DONE")
