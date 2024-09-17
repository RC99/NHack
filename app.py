from flask import Flask, render_template, send_from_directory
import os
import visuals  # Import the visuals module

app = Flask(__name__)

# Define the directory for saving images
image_dir = '/Users/reetvikchatterjee/NvidiaHack/images'

@app.route('/')
def index():
    visuals.setup_images(image_dir)  # Generate images using the visuals module
    num_images = len([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
    return render_template('index.html', num_images=num_images)

@app.route('/Users/reetvikchatterjee/NvidiaHack/images/<filename>')
def images(filename):
    return send_from_directory(image_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
