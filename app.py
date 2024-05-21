import os
import torch
import gdown
import streamlit as st
from transformers import AutoTokenizer, AutoTokenizer

def load_model(gdrive_id='10IOylWbThT3M-pC6eunjh_Z0dc_VfQ-V'):

  model_path = 'xlm-roberta-base-squad2'
  if not os.path.exists(model_path):
    # download folder
    gdown.download_folder(id=gdrive_id)
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoTokenizer.from_pretrained(model_path)
  return tokenizer, model

tokenizer, model = load_model()

def inference(question, context, tokenizer, model):
  with torch.no_grad():
    input = tokenizer.encode_plus(context, question, return_tensors='pt')
    res = model(**input)
    start_position = torch.argmax(res.start_logits[0])
    end_position = torch.argmax(res.end_logits[0])
    answer = tokenizer.decode(input['input_ids'][0][start_position:end_position+1], skip_special_tokens=True)
  return answer

def main():
  st.title('Extractive Question Answering - Machine Reading Comprehension')
  st.title('Model: XLM-RoBERTa-Base. Dataset: SQUAD-V2')
  context = st.text_input("Context: ", "Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam. Hà Nội nằm về phía tây bắc của trung tâm vùng đồng bằng châu thổ sông Hồng, với địa hình bao gồm vùng đồng bằng trung tâm và vùng đồi núi ở phía bắc và phía tây thành phố.")
  question = st.text_input("Question: ", "Thủ đô của nước Việt Nam là gì?")
  result = inference(question, context, tokenizer, model)
  st.success(result) 

if __name__ == '__main__':
     main() 
