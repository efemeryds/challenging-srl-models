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
bert_model = 'structured-prediction-srl-bert'
basic_model = 'structured-prediction-srl'

srl_predictor = load_predictor(basic_model)

""" """














