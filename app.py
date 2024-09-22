import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Model, GPT2Config, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Load the model and tokenizer from Hugging Face
# @st.cache_resource
# def load_model():
#     model_name = "Nishantc05/emailSubGen-bartmodel"  # Replace with your Hugging Face model
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = BartForConditionalGeneration.from_pretrained(model_name)
#     return tokenizer, model

# # Load the model
# tokenizer_hub, model = load_model()

# Set up the Streamlit app
st.title("Email Subject Line Generator")

# Get the email content from the user
email_content = st.text_area("Enter Email Content:", height=200)

# Only display the generated subject line when the button is pressed
if st.button("Generate Subject Line"):
    if email_content:  # Check if the user entered any email content
        # Tokenize the input using the tokenizer, not the model
        model_name = "Nishantc05/emailSubGen-bartmodel"  # Replace with your Hugging Face model
        tokenizer_hub = AutoTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure the model and tensors are sent to the device (CPU in this case)
        model = model.to(device)
        inputs = tokenizer_hub(email_content, return_tensors="pt", max_length=512, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        
        # Generate the subject line using the model
        outputs = model.generate(inputs['input_ids'], max_length=100, num_beams=5, early_stopping=True)
        
        # Decode and clean the output (skipping special tokens)
        subject_line = tokenizer_hub.decode(outputs[0], skip_special_tokens=True)

        # Remove potential custom tokens manually if they persist
        # unwanted_tokens = ['<ANSWER_ENDED>', '<QUESTION>', '<ANSWER>']
        # for token in unwanted_tokens:
        #     subject_line = subject_line.replace(token, '')
        
        # Display only the generated subject line (answer)
        st.subheader("Generated Subject Line:")
        st.write(subject_line)
    else:
        # Prompt the user to enter something if the text box is empty
        st.write("Please enter some email content to generate a subject line.")

