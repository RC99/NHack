from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load BlenderBot model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

# Chatbot function
def chat_blenderbot(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply

# Test the chatbot
print(chat_blenderbot("How many traffic accidents occured in Maryland in past 2 years?"))
