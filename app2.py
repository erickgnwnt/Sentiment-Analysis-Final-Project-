from flask import Flask, request, jsonify,render_template
import torch
from transformers import RobertaTokenizer,RobertaForSequenceClassification
# from transformers import BertTokenizer,BertForSequenceClassification
app = Flask(__name__)

model_path = 'classifier1robertafinetuned1e5ep6.pt'
# model_path = 'classifier1bertfinetuned.pt'
device = torch.device('cpu')  # Use CPU for inference

# Load the model RoBERTa
model = RobertaForSequenceClassification.from_pretrained('w11wo/indonesian-roberta-base-sentiment-classifier' )
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
tokenizer = RobertaTokenizer.from_pretrained('w11wo/indonesian-roberta-base-sentiment-classifier')


#BERT
# model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1' )
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()
# tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')


# @app.route('/')


# def home():
#     return render_template('index.html')

# @app.route('/analyze_sentiment', methods=['POST'])
# def analyze_sentiment():
#     try:
#         # Get the input text from the HTML form
#         data = request.form['text']

#         # Tokenize the input text and convert it to IDs
#         inputs = tokenizer(data, return_tensors='pt', truncation=True, padding=True)

#         # Move the input tensors to the CPU (if needed)
#         inputs = {key: val.to(device) for key, val in inputs.items()}

#         # Perform sentiment analysis
#         with torch.no_grad():  # To disable gradient computation during inference
#             outputs = model(**inputs)

#         # Get the predicted label
#         prediction = torch.argmax(outputs.logits, dim=1).item()
#         #persentase
#         # softmax = torch.nn.Softmax(dim=1)
#         # probabilities = softmax(predictions)
#         # positive_percentage = probabilities[:, 1].item() * 100
#         # negative_percentage = probabilities[:, 0].item() * 100
#         # Map the label to a sentiment string
#         sentiment = None
#         if prediction == 0:
#             sentiment = "positive"
#         elif prediction == 1:
#             sentiment = "neutral"
#         elif prediction == 2:
#             sentiment = "negative"

#         return jsonify({'sentiment': sentiment})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

#########batas

# Function to calculate the percentage of sentiment prediction
def calculate_percentage(predictions):
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(predictions)
    positive_percentage = probabilities[:, 0].item() * 100
    negative_percentage = probabilities[:, 2].item() * 100
    neutral_percentage = probabilities[:, 1].item() * 100
    return positive_percentage, negative_percentage, neutral_percentage

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            model_output = model(**encoded_input).logits
        positive_percentage, negative_percentage, neutral_percentage = calculate_percentage(model_output)
        return render_template("result.html", text=text, positive=positive_percentage, negative=negative_percentage, neutral=neutral_percentage)
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)