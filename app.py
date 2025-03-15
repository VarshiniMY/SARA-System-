from flask import Flask, jsonify, render_template, redirect, url_for, request, flash
from werkzeug.utils import secure_filename
import secrets, os

import numpy as np
import cv2

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route("/")
def index():
    return render_template('/index.html')

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return redirect("/")
    
    file = request.files['file']
    print(file)
    print(file.filename)
    f = open("demofile2.txt", "w")
    f.write(file.filename)
    f.close()
    import subprocess
    scripts = [ "camera_video.py","mainspeed.py"]
    processes = []
    for script in scripts:
        p = subprocess.Popen(["python", script])
        processes.append(p)
    for p in processes:
        p.wait()

    return redirect('/')

if __name__ == ' __main__ ':
    app.run()
