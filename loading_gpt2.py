from transformers import GPT2LMHeadModel, GPT2Tokenizer

#https://huggingface.co/transformers/v3.0.2/model_doc/gpt2.html

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

model.save_pretrained('./gpt2_model/')
tokenizer.save_pretrained('./gpt2_model/')

