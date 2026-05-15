from transformers import pipeline
gen=pipeline("text-generation", model="gpt2")
result=gen("the future of AI in India is",max_new_tokens=50,num_return_sequences=1)
print(result[0]['generated_text'])
creative=gen("once upon a time in mandya",max_new_tokens=50,temperature=0.1,do_sample=True)
focused=gen("once upon a time in mandya",max_new_tokens=50,temperature=1.2,do_sample=True)

print("Creative:",creative[0]['generated_text'])
print("focused:",focused[0]['generated_text'])