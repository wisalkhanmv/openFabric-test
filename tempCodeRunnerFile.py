def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    
    #load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    output = []
    print(request.text)
    for text in request.text:
        try:
            input_ids = tokenizer.batch_encode_plus([text], return_tensors = "pt")["input_ids"]
            # input_ids = input_ids.unsqueeze(0)
            print(input_ids)
            response = model.generate(input_ids, 
                                max_length=MAX_LENGTH, 
                                top_p=0.9, 
                                top_k=40)["generated_text"]
            response = response.squeeze(0)
            print(response)
            response = tokenizer.batch_decode(response, skip_special_tokens=True)
            output.append(response)

        except Exception as e:
            print("Error occurred: ", e)

    return SimpleText(dict(text=output))
