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
srl_predictor = load_predictor('structured-prediction-srl-bert')
output = srl_predictor.predict("The killer killed the victim with a knife.")

print('DONE')






