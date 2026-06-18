# 🧠 Semantic Analyzer - Emotion Detection using NLP and Machine Learning

A lightweight and interactive **Emotion Detection Web Application** built using **Natural Language Processing (NLP)**, **Scikit-Learn**, and **Streamlit**. The model is trained on 16,000 labeled sentences and predicts the underlying emotion expressed in user-input text.

---

## 📌 Features

* 🔍 Detects emotions from natural language text.
* ⚡ Fast inference using TF-IDF + Machine Learning.
* 🎨 Modern and interactive Streamlit UI.
* 💾 Pre-trained model saved using Pickle (.pkl).
* 📊 Displays predicted emotion and confidence scores.
* 🚀 Easy deployment on Streamlit Cloud or locally.
* 🧩 Modular project structure for future improvements.

---

## Supported Emotions

The model currently recognizes:

* 😊 Joy
* 😢 Sadness
* ❤️ Love
* 😡 Anger
* 😨 Fear
* 😲 Surprise

---

## Project Structure

```text
Semantic-Analyser/
│
├── train.txt                # Dataset containing sentences and labels
├── train_model.ipynb        # Google Colab notebook used for training
├── sentiment_model.pkl      # Trained machine learning model
├── vectorizer.pkl           # TF-IDF vectorizer
├── labels.txt               # Emotion labels
├── app.py                   # Streamlit web application
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── .gitignore
```

---

## Dataset

The dataset contains approximately **16,000 text samples** with corresponding emotion labels.

Example:

```text
I am feeling very happy today ; joy
I feel lonely and depressed ; sadness
I am furious about this ; anger
```

---

## Model Pipeline

### Text Preprocessing

* Lowercasing
* Tokenization
* Stopword removal
* TF-IDF vectorization

### Machine Learning Model

* Logistic Regression / LinearSVC
* Trained on 80% of the dataset
* Tested on 20% of the dataset

---

## Tech Stack

### Backend

* Python

### Machine Learning

* Scikit-Learn
* TF-IDF Vectorizer
* Logistic Regression / LinearSVC

### Frontend

* Streamlit

### Data Handling

* Pandas
* NumPy

---

## Installation

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Semantic-Analyser.git

cd Semantic-Analyser
```

---

### Create a virtual environment

#### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The application will open automatically in your browser:

```text
http://localhost:8501
```

---

## Example

### Input

```text
I am extremely excited about my internship.
```

### Output

```text
Emotion Detected: Joy 😄
```

---

## Model Saving

The trained model and vectorizer are stored as pickle files:

```python
pickle.dump(model, open("sentiment_model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))
```

Load them using:

```python
with open("sentiment_model.pkl","rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)
```

---

## Dependencies

* streamlit
* scikit-learn
* pandas
* numpy

Install them via:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

* 🤖 Fine-tune BERT or DistilBERT for higher accuracy.
* 📈 Add sentiment confidence visualization.
* 📝 Emotion history tracking.
* 🌐 Multi-language support.
* ☁️ Streamlit Cloud deployment.
* 📊 Analytics dashboard.
* 🎭 Emoji-based emotion visualization.
* 🔥 Dark mode and glassmorphism UI.

---

## Applications

* Mental health monitoring
* Chatbots and conversational AI
* Customer feedback analysis
* Social media sentiment analysis
* Product review classification
* Human-computer interaction

---

## Author

### Suteekshn Manchanda

B.Tech Electronics and Communication Engineering |
Machine Learning Engineer | NLP | Computer Vision | AI

---

---

## ⭐ If you found this project useful, please consider giving it a star on GitHub!
