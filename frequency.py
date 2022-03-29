""" Evaluation of the results """

import pandas as pd


def count_failure(input_data):
    count = (((input_data['eval'] == 0).sum() + 0.0001) / len(results_arg1)) * 100
    count = round(count, 2)
    return count


def get_evaluation(input_data):
    tmp_list = []
    frequent = input_data[input_data['if_frequent'] == 1]
    non_frequent = input_data[input_data['if_frequent'] == 0]
    tmp_list.append({"type": "general", "failure_rate (%)": count_failure(input_data)})
    tmp_list.append({"type": "frequent", "failure_rate (%)": count_failure(frequent)})
    tmp_list.append({"type": "non_frequent", "failure_rate (%)": count_failure(non_frequent)})
    return


results_arg1 = pd.read_csv("results/frequency_arg1.csv")
final_arg1 = pd.DataFrame(get_evaluation(results_arg1))
final_arg1.to_csv("evaluation/frequency_arg1.csv", index=False)

results_argm_loc = pd.read_csv("results/frequency_argm_loc.csv")
final_argm_loc = pd.DataFrame(get_evaluation(results_arg1))
final_argm_loc.to_csv("evaluation/frequency_argm_loc.csv", index=False)

print("DONE")
