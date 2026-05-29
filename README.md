# 🧠 Early Depression Detection System

A multimodal machine learning web application that detects signs of depression through **text**, **facial expressions (image)**, and **voice (audio)** inputs — built with Flask and TensorFlow.

---

## 📌 About the Project

Depression is one of the most underdiagnosed mental health conditions. This project aims to assist early detection by analyzing three different input modalities:

- 💬 **Text** — Analyzes what the user writes using a trained ML classifier
- 🖼️ **Image** — Analyzes facial expressions using a CNN model trained on the CK+48 dataset
- 🎙️ **Audio** — Analyzes voice patterns using MFCC and Chroma features via a deep learning model

The final prediction combines all three signals to give an overall result: **Depressed**, **Possibly Depressed**, or **Not Depressed**.

---

## 🖥️ Demo

> Run locally and visit `http://127.0.0.1:5000`

---

## 🗂️ Project Structure

```
Early-Depression-Detection-main/
│
├── web_app/
│   ├── models/
│   │   ├── audio_model.h5       # Trained audio model
│   │   ├── image_model.h5       # Trained image model
│   │   ├── text_model.pkl       # Trained text classifier
│   │   └── vectorizer.pkl       # Text vectorizer
│   │
│   ├── static/
│   │   ├── css/
│   │   ├── fonts/
│   │   ├── img/
│   │   └── js/
│   │
│   ├── templates/
│   │   ├── index.html
│   │   ├── donate.html          # Main input page
│   │   └── result.html          # Result display page
│   │
│   ├── app.py                   # Main Flask application
│   └── spectrogram.py           # Audio spectrogram helper
│
├── image_model.ipynb            # Image model training notebook
├── audio_model.ipynb            # Audio model training notebook
├── text_model.ipynb             # Text model training notebook
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

### Text Analysis
- User input is vectorized using a pre-trained `TfidfVectorizer`
- A trained classifier predicts depression probability
- Thresholds: `≥ 0.75` → Depressed, `≥ 0.55` → Possibly Depressed

### Image Analysis
- Facial image is resized to `48x48` and passed through a CNN
- Model predicts emotional state from facial expression
- Trained on the **CK+48** dataset

### Audio Analysis
- Audio is loaded with `librosa` (3 seconds, offset 0.5s)
- Features extracted: **MFCC (40)** + **Chroma (12)** = 52 features
- Deep learning model predicts depression likelihood from voice patterns

### Final Decision Logic
1. If text says **Depressed** → Final: Depressed
2. Else audio result is the primary signal
3. Text weak signal used as fallback
4. Image used as supporting context

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/meghana-reddy-15/early-depression-detection.git
cd early-depression-detection/web_app

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open your browser and go to: `http://127.0.0.1:5000`

---

## 📦 Requirements

See `requirements.txt` for full list. Key dependencies:

| Package | Purpose |
|---|---|
| Flask | Web framework |
| TensorFlow | Image & Audio models |
| scikit-learn | Text classification |
| librosa | Audio feature extraction |
| numpy | Numerical processing |
| Pillow | Image preprocessing |

---

## 📊 Models Used

| Modality | Model Type | Dataset |
|---|---|---|
| Text | TF-IDF + ML Classifier | Custom |
| Image | CNN (Conv2D) | CK+48 |
| Audio | Deep Learning (MFCC+Chroma) | Custom |

---

## ⚠️ Disclaimer

This tool is **not a medical diagnosis**. It is an academic project built for educational purposes. If you or someone you know is struggling with depression, please consult a licensed mental health professional.

**Helpline (India):** iCall — 9152987821

---

## 👩‍💻 Author

**Meghana Reddy**
- GitHub: [@meghana-reddy-15](https://github.com/meghana-reddy-15)
- Email: meghanareddy.k2006@gmail.com
- Bengaluru, Karnataka, India
