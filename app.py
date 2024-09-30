from flask import Flask, render_template, send_from_directory, request, jsonify
import os
from openai import OpenAI 
import visuals 
from gemini import google_search  

app = Flask(__name__)

client = OpenAI(api_key="")

image_dir = '/Users/reetvikchatterjee/NvidiaHack/images'

@app.route('/')
def index():
    visuals.setup_images(image_dir) 
    num_images = len([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
    images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    return render_template('index.html', images=images, num_images=num_images)

@app.route('/images/<filename>')
def images(filename):
    return send_from_directory(image_dir, filename)

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/ask_gpt', methods=['POST'])
def ask_gpt():
    user_input = request.json.get('prompt')
    additional_info = google_search(user_input)  
    response = generate_gpt_response(user_input, additional_info)  
    return jsonify({'response': response})

def generate_gpt_response(prompt, additional_info):
    combined_prompt = f"{prompt}\n\nAdditional Information:\n" + "\n".join(additional_info)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": combined_prompt}
            ]
        )
        return response.choices[0].message.content  
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
