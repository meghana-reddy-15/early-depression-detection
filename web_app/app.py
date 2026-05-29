import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import librosa

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# LOAD MODELS
# -----------------------------
text_model = pickle.load(open("models/text_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

image_model = load_model("models/image_model.h5")
audio_model = load_model("models/audio_model.h5")

# -----------------------------
# CONFIG
# -----------------------------
UPLOAD_FOLDER = 'static/audio_uploads'
IMAGE_FOLDER = 'static/image_uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# -----------------------------
# TEXT PREDICTION (FIXED)
# -----------------------------
def predict_text(user_text):
    text_vector = vectorizer.transform([user_text])
    prob = text_model.predict_proba(text_vector)[0][1]

    print("Text Probability:", prob)

    # 🔥 stricter thresholds (important)
    if prob >= 0.75:
        return "Depressed"
    elif prob >= 0.55:
        return "Possibly Depressed"
    else:
        return "Not Depressed"

# -----------------------------
# IMAGE PREDICTION (CONTROLLED)
def predict_image(img_path):

    img = image.load_img(img_path, target_size=(48, 48))
    img_array = image.img_to_array(img) / 255.0
    img_array = img_array.reshape(1, 48, 48, 3)

    pred = image_model.predict(img_array)[0]

    print("Image Prediction:", pred)

    depressed_prob = pred[0]
    not_dep_prob = pred[1]

    confidence = max(depressed_prob, not_dep_prob)

    # 🔥 CORRECT LOGIC
    if depressed_prob > 0.7:
        return f"Depressed ({confidence:.2f})"
    elif depressed_prob > 0.5:
        return f"Possibly Depressed ({confidence:.2f})"
    else:
        return f"Not Depressed ({confidence:.2f})"

# -----------------------------
# AUDIO PREDICTION (FINAL)
# -----------------------------
def predict_audio(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)

    # Feature extraction
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)

    features = np.hstack([mfcc, chroma])
    features = features.reshape(1, -1)

    pred = audio_model.predict(features)[0]
    print("Audio Prediction:", pred)

    depressed_prob = pred[0]
    not_dep_prob = pred[1]

    confidence = float(np.max(pred))

    # ✅ CORRECT DECISION
    if depressed_prob > not_dep_prob:
        label = "Depressed"
    else:
        label = "Not Depressed"

    # ✅ Confidence handling
    if label == "Depressed" and confidence < 0.6:
        return f"Possibly Depressed ({confidence:.2f})"
    else:
        return f"{label} ({confidence:.2f})"
# -----------------------------
# FINAL DECISION (CORRECT LOGIC)
# -----------------------------
def combine_results(text_result, image_result, audio_result):

    # 🔴 TEXT (only if strong)
    if text_result == "Depressed":
        return "Depressed"

    # 🔥 AUDIO (main decision if text is weak)
    if audio_result:
      if audio_result.startswith("Depressed"):
        return "Depressed"
      elif audio_result.startswith("Possibly"):
        return "Possibly Depressed"
      elif audio_result.startswith("Not"):
        return "Not Depressed"

    # 🔸 TEXT weak case
    if text_result == "Possibly Depressed":
        return "Possibly Depressed"

    # ❌ IMAGE ignored
    return "Not Depressed"

# -----------------------------
# ROUTES
# -----------------------------
from flask import redirect

@app.route('/')
def index():
    return redirect('/donate')


@app.route('/donate', methods=['GET', 'POST'])
def donate():
    if request.method == 'POST':

        text_result = None
        image_result = None
        audio_result = None

        
        # -----------------------------
# TEXT (FIXED LOCATION)
# -----------------------------
        user_text = request.form.get("user_text")

        if user_text and user_text.strip() != "":
               text_result = predict_text(user_text)
        else:
             text_result = None
        # -----------------------------
        # IMAGE
        # -----------------------------
        image_file = request.files.get('image')

        if image_file and image_file.filename != "":
            img_path = os.path.join(app.config['IMAGE_FOLDER'], secure_filename(image_file.filename))
            image_file.save(img_path)
            image_result = predict_image(img_path)

        # -----------------------------
        # AUDIO
        # -----------------------------
        audio_file = request.files.get('file')

        if audio_file and audio_file.filename != "":
            wav_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
            audio_file.save(wav_path)
            audio_result = predict_audio(wav_path)

        # -----------------------------
        # FINAL RESULT
        # -----------------------------
        final_result = combine_results(text_result, image_result, audio_result)

        return render_template(
            "result.html",
            result=final_result,
            text_result=text_result,
            image_result=image_result,
            audio_result=audio_result
        )

    return render_template('donate.html')


if __name__ == '__main__':
    app.run(debug=True)