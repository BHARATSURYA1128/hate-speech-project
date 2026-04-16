from flask import Flask, request, jsonify, send_file
import pickle

app = Flask(__name__)

with open('../model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    return jsonify({"prediction": int(pred)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
