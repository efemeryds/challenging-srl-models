from srl_tests.frequency.ARG1 import run_arg1_test

input_sentence = "Someone stole {thing} from my grandfather's house yesterday evening."

print("Runs arg1 test for common nouns..")
common_nouns = ["apple", "watch", "guitar", "glasses", "computer", "plant", "ants", "fridge", "stolen car",
                "stolen music", "stolen art"]
run_arg1_test(input_sentence, common_nouns)


print("Runs arg1 test for uncommon nouns..")
uncommon_nouns = ["genipap", "futhorc", "witenagemot", "gossypol", "chaulmoogra", "brummagem", "alsike",
                  "chersonese", "cacomistle", "yogh", "smaragd", "duvetyn", "pyknic", "fylfot", "yataghan",
                  "dasyure", "simoom", "stibnite", "kalian"]
run_arg1_test(input_sentence, uncommon_nouns)





