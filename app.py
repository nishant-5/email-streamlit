import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Model, GPT2Config, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.title("Subject Line Email Generation")

# Text input from user
email_content = st.text_area("Enter the email content", height=200)

if st.button("Generate a subject line"):
    if email_content:
        # Tokenize and generate subject line
        model = AutoModelForSeq2SeqLM.from_pretrained("Nishantc05/emailSubGen-bartmodel")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        tokenizer = AutoModelForSeq2SeqLM.from_pretrained("Nishantc05/emailSubGen-bartmodel")
        #tokenizer.padding_side = "left"
        #tokenizer.pad_token=tokenizer.eos_token
        inputs = tokenizer.encode(email_content, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=100, num_beams=5, early_stopping=True)
        subject_line = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        unwanted_tokens = ['<ANSWER_ENDED>', email_content,'Answer']
        for token in unwanted_tokens:
            subject_line = subject_line.replace(token, '')
        #subject_line = tokenizer.decode(skip_special_tokens=True)
        st.subheader("Generated Subject line.")
        st.write(subject_line)
    else:
        st.write("Please enter some email content to generate a subject line.")
