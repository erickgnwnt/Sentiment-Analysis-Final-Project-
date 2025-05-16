from flask import Flask, render_template, request
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
import torch

# model_name = "modelrobertajson.JSON"  # pcnya meledak
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']

    # Perform sentiment analysis using your RoBERTa model
    sentiment = perform_sentiment_analysis(text)

    return render_template('result.html', sentiment=sentiment)

# def perform_sentiment_analysis(text):
#     # Tokenize the input text
#     encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
#     input_ids = encoded_input["input_ids"].to(device)
#     attention_mask = encoded_input["attention_mask"].to(device)

#     # Forward pass through the model
#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask)

#     # Get predicted sentiment
#     logits = outputs.logits
#     probabilities = torch.softmax(logits, dim=1).squeeze()
#     sentiment_id = torch.argmax(probabilities).item()

#     # Map sentiment id to label
#     sentiment_labels = ["Negative", "Neutral", "Positive"]
#     sentiment = sentiment_labels[sentiment_id]

#     return sentiment


if __name__ == '__main__':
    app.run(debug=True)