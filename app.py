from flask import Flask, request, jsonify
import pandas as pd
from modelHandler import ModelHandler

app = Flask(__name__)

url = 'https://drive.google.com/file/d/1kqnB4J8FuF1k8xLIvfbPE1jsqUwd3wVH/view?usp=sharing'
model_handler = ModelHandler(url)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get JSON data from the request
        input_df = pd.DataFrame(data)  # Convert JSON data to DataFrame
        predictions = model_handler.predict(input_df)  # Get predictions
        return jsonify(predictions)  # Return predictions as JSON
    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Handle errors gracefully



@app.route('/model_performance', methods=['GET'])
def model_performance():
    performance = model_handler.evaluate_models()
    return jsonify({"RMSE": performance})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
