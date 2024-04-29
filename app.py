from flask import Flask, request, jsonify
from flask_cors import CORS
from named_entity_recognition.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.ner = PredictionPipeline()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        return jsonify(cl_app.ner.predict(data['sentence']))
    except Exception as e:
        return ({"message": str(e)}, 400)
    
if __name__ == "__main__":
    cl_app = ClientApp()
    app.run(debug=True)