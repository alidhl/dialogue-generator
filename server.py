"""Basic server for the API using Flask."""
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import Model

app = Flask(__name__)
CORS(app)
model = Model()

@app.route('/generate', methods=['POST'])
def generate_text():
    seed_text = request.json['seed']
    generated_text = model.predict(seed_text, 50)
    return jsonify({'predicted_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)