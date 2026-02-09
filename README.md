📊 Sentiment Analysis using NLTK & HuggingFace Transformers

This project performs Sentiment Analysis on textual data using Natural Language Processing (NLP) techniques.
It combines traditional NLP preprocessing with NLTK and deep learning-based transformer models from HuggingFace.

The project was developed and tested on Kaggle Notebook environment.

🚀 Features

Text preprocessing using NLTK

Stopword removal and tokenization

Sentiment classification using HuggingFace Transformers

Pretrained model from cardiffnlp/twitter-roberta-base-sentiment

Data visualization using Matplotlib

Performance analysis and result visualization

🛠️ Technologies Used

Python

NLTK

HuggingFace Transformers

PyTorch

Matplotlib

Pandas

NumPy

Kaggle Notebook

📂 Project Structure
sentiment-analysis/
│
├── dataset/                # Dataset files (if included)
├── notebook.ipynb          # Kaggle notebook
├── outputs                 # outputs
└── README.md               # Project documentation

📌 Model Used

This project uses the pretrained transformer model:

cardiffnlp/twitter-roberta-base-sentiment

Based on RoBERTa architecture

Fine-tuned for sentiment classification

Outputs: Positive, Negative, Neutral

🔍 Workflow

Load dataset

Text cleaning and preprocessing (NLTK)

Tokenization using HuggingFace tokenizer

Sentiment prediction using pretrained transformer

Visualization of results using Matplotlib

📊 Visualization

Sentiment distribution bar chart

Confidence score comparison

Prediction frequency analysis

📦 Installation

Clone the repository:

git clone https://github.com/Nilam474/sentiment-analysis.git
cd sentiment-analysis


Install required libraries:

pip install -r requirements.txt


If running on Kaggle, most dependencies are pre-installed.

▶️ How to Run

Open the Kaggle notebook

Upload dataset (if required)

Run all cells sequentially

View predictions and visualizations

🧠 Learning Outcomes

Understanding NLP preprocessing

Working with pretrained transformer models

Applying sentiment classification

Data visualization for model results

📈 Future Improvements

Fine-tuning the transformer model

Adding confusion matrix and classification report

Deploying using Flask / FastAPI

Creating a simple web interface
