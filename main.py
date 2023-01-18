import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the maximum length of the generated text
MAX_LENGTH = 100

############################################################
# Callback function called on update config
############################################################

def config(configuration: ConfigClass):
    # Load configuration
    configuration.load()

    # Set the maximum length of the generated text
    global MAX_LENGTH
    MAX_LENGTH = configuration.get("max_length", MAX_LENGTH)


############################################################
# Callback function called on each execution pass
############################################################

def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    
    #load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    output = []
    chat_history_ids = torch.tensor([])
    print(request.text)
    for text in request.text:
        try:
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if len(chat_history_ids) > 0 else new_user_input_ids

            # generated a response while limiting the total chat history to 1000 tokens, 
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

            # pretty print last ouput tokens from bot
            output.append(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))

        except Exception as e:
            print("Error occurred: ", e)

    return SimpleText(dict(text=output))
