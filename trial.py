import os
#from gtts import gTTS 
from pathlib import Path
import appdirs
import gdown
import torch
import logging
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from summarizer import Summarizer
#from summarizer import TransformerSummarizer

class BartSumSummarizer():
    def __init__(self, state_dict_key='model', pretrained="facebook/bart-large-cnn"):
        bart = BartForConditionalGeneration.from_pretrained(pretrained)
        tokenizer = BartTokenizer.from_pretrained(pretrained)
        self.tokenizer = tokenizer
        self.bart = bart

    def summarize_string(self, source_line, min_length=55, max_length=300):
        source_line = [source_line]
        inputs = self.tokenizer.batch_encode_plus(source_line, max_length=1024, return_tensors='pt')
        # Generate Summary
        summary_ids = self.bart.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], num_beams=4, min_length=min_length, max_length=max_length)
        return [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
       

def bart(contents):
    document = str(contents)
    doc_length = len(document.split())
    min_length = int(doc_length/6)
    max_length = min_length+200
    transcript_summarized = BartSumSummarizer().summarize_string(document, min_length=min_length, max_length=max_length)
    return transcript_summarized

def bert(contents):
    bert_model = Summarizer()
    bert_summary = ''.join(bert_model(contents, min_length=60))
    return bert_summary
    
'''def gpt2(contents):
    GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
    full = ''.join(GPT2_model(contents))
    return full'''
body = '''
        Eight people are dead following two shootings at shisha bars in the western German city of Hanau. At least five people were injured after gunmen opened fire at about 22:00 local time (21:00 GMT), police told the BBC. Police added that they are searching for the suspects, who fled the scene and are currently at large. The first shooting was at a bar in the city centre, while the second was in Hanau's Kesselstadt neighbourhood, according to local reports. Police officers and helicopters are patrolling both areas. An unknown number of gunmen killed three people at the first shisha bar, Midnight, before driving to the Arena Bar & Cafe and shooting dead another five victims, regional broadcaster Hessenschau reports. A dark-coloured vehicle was then seen leaving the scene.The motive for the attack is unclear, a police statement said. Can-Luca Frisenna, who works at a kiosk at the scene of one of the shootings said his father and brother were in the area when the attack took place. It's like being in a film, it's like a bad joke, that someone is playing a joke on us, he told Reuters.I can't grasp yet everything that has happened. My colleagues, all my colleagues, they are like my family - they can't understand it either. Hanau, in the state of Hessen, is about 25km (15 miles) east of Frankfurt. It comes four days after another shooting in Berlin, near a Turkish comedy show at the Tempodrom concert venue, which killed one person.
        '''
bar=bart(body)
print("BART MODEL SUMMARY\n"+bar)
print('\n')
print("BERT MODEL SUMMARY\n"+bert(body))
#print('\n')
#print("GPT2 MODEL SUMMARY\n"+gpt2(body))
language = 'en'
myobj = gTTS(text=bar, lang=language, slow=False)  
myobj.save("summary.mp3") 
