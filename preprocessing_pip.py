import re
import pickle
import torch
from transformers import DistilBertTokenizerFast

class CVPreprocessor:
    """
    Preprocessing pipeline untuk CV/Resume text sebelum prediksi.
    Melakukan cleaning, tokenization, dan encoding sesuai dengan proses training.
    """
    
    def __init__(self, tokenizer_path='distilbert-base-uncased', label_encoder_path='label_encoder.pkl'):

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
        
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.contractions = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot", 
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he's": "he is", "she's": "she is",
            "it's": "it is", "I'm": "I am", "isn't": "is not",
            "let's": "let us", "mightn't": "might not", "mustn't": "must not",
            "shan't": "shall not", "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they're": "they are", "wasn't": "was not",
            "we're": "we are", "weren't": "were not", "what's": "what is",
            "won't": "will not", "wouldn't": "would not", "you're": "you are",
            "you've": "you have"
        }
        
        self.max_length = 512
    
    def expand_contractions(self, text):
        pattern = re.compile(r'\b(' + '|'.join(self.contractions.keys()) + r')\b')
        return pattern.sub(lambda x: self.contractions[x.group()], text)
    
    def clean_text(self, text):
        text = str(text).lower()
        text = self.expand_contractions(text)
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def preprocess_for_prediction(self, ocr_text):
        clean_text = self.clean_text(ocr_text)
        
        tokens = self.tokenizer(
            clean_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        return tokens
    
    def decode_prediction(self, label_id):
        return self.label_encoder.inverse_transform([label_id])[0]
    
    def get_categories(self):
        return list(self.label_encoder.classes_)