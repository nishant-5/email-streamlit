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
        tokenizer = AutoModelForSeq2SeqLM.from_pretrained("Nishantc05/emailSubGen-bartmodel")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        inputs = tokenizer(email_content, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate the subject line using the model
        outputs = model.generate(inputs['input_ids'], max_length=20, num_beams=5, early_stopping=True)
        
        # Decode and clean the output (skipping special tokens)
        subject_line = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Remove potential custom tokens manually if they persist
        unwanted_tokens = ['<ANSWER_ENDED>', '<QUESTION>', '<ANSWER>']
        for token in unwanted_tokens:
            subject_line = subject_line.replace(token, '')
        st.subheader("Generated Subject line.")
        st.write(subject_line)
    else:
        st.write("Please enter some email content to generate a subject line.")
