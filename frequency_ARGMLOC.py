from srl_tests.frequency.ARGM_LOC import run_location_test

input_sentence = "When I was younger someone told me about a magical place, this happened {location}."

print("Runs common tests with ARGM-LOC detection task..")
common_words = ["far away", "in the Wonderland", "next to my home", "on the street", "in the forest", "in the hospital"]
run_location_test(input_sentence, common_words)

print("Runs uncommon tests with ARGM-LOC detection task..")
uncommon_words = ["in the bastion", "in my dreams"]
run_location_test(input_sentence, uncommon_words)

