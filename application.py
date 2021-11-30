from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import shutil
import time
from breathe_net import *

image_path = "static/images/"

application = Flask(__name__)
application.debug=True
# Ensure static/images directory exists for image xray uploads
os.makedirs(os.path.join("static/images"), exist_ok=True)

def get_time():
    current_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    return current_time

# Delete all contents of a directory
# https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
def clear_contents(image_path):
    # DELETE IMAGE DIR CONTENTS
    for the_file in os.listdir(image_path):
        file_path = os.path.join(image_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

### PAGE DIRECTS

@application.route("/home")
@application.route("/")
def home():
    return render_template("home.html")

@application.route("/about")
def about():
    return render_template("about.html", title="About")

@application.route("/covid")
def covid():
    return render_template("covid.html", title="COVID-19")


@application.route("/diagnose")
def diagnose():
    return render_template("diagnose.html", title="Diagnose")

@application.route("/resources")
def resources():
    return render_template("resources.html", title="Resources")




# Render results.html with diagnosis/confidence data output
@application.route('/results', methods=['GET', 'POST'])
def results():
    # Delete contents of Images folder to save server space
    clear_contents("static/images")

    # File upload
    if request.method == 'POST':
        f = request.files['file']

        current_time = get_time()

        # Create directory for current image
        os.makedirs(os.path.join('static/images', current_time))

        # Save uploaded image to directory with current time & name info
        img_path = os.path.join('static/images', current_time, secure_filename(f.filename))
        f.save(img_path)

        # Call neural net to run prediction batch of 1 image to get diagnoses & confidence values
        label, confidence = predict_img(model, img_path)

        has_cap = False

        if label == "PNEUMONIA":
            has_cap = True

        # Render results.html with diagnoses information as inputs to display
        return render_template("results.html", title="Results", diagnoses="{} with {:.2f}% Confidence.".format(label,confidence), img_path=img_path, has_cap=has_cap)

if __name__ == "__main__":
    application.run()
