from flask import Flask, request, render_template, redirect, url_for
from app import app, APP_ROOT

from app.process import prognosis

import os


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        f = request.files['audio_data']
        with open('app/audio.wav', 'wb+') as audio:
            f.save(audio)

        return redirect(url_for('predict'))
    else:
        return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    features = []
    for key in request.form:
        features.append(request.form[key])
    x = prognosis('audio.wav', features)
    # print(x)
    return render_template("output.html", result=x)


if __name__ == "__main__":
    app.run(debug=True)
