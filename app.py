from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from modelHandler import ModelHandler
from modelTrainer import ModelTrainer

app = Flask(__name__)



###this api will get us product, based on popular influencer feedback
@app.route('/popular-goods', methods=['GET'])
def popular(file_path = None):
    file_path = file_path or './data-sets/amazon_reviews.csv'
    models = {
        'LogisticsRegression': LogisticRegression(random_state=42, max_iter=1000),
        'GradientBooster': HistGradientBoostingClassifier(max_iter=100, random_state=42)
    }
    trainer = ModelTrainer(models=models, file_path=file_path)
    trainer.load_and_preprocess_data()
    trainer.train_models()
    bp = trainer.best_performer()
    top_5_recommendations = trainer.recommend_top_5_products()
    # print(top_5_recommendations)
    # Convert the DataFrame to a list of dictionaries
    result = top_5_recommendations.to_dict(orient='records')

    # Return the JSON response
    return jsonify(result)
#    return jsonify(top_5_recommendations.to_string(index=False))

@app.route('/predict', methods=['POST'])
def predict():
    url = 'https://drive.google.com/file/d/1kqnB4J8FuF1k8xLIvfbPE1jsqUwd3wVH/view?usp=sharing'
    model_handler = ModelHandler(url)
    try:
        data = request.get_json()  # Get JSON data from the request
        input_df = pd.DataFrame(data)  # Convert JSON data to DataFrame
        predictions = model_handler.predict(input_df)  # Get predictions
        return jsonify(predictions)  # Return predictions as JSON
    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Handle errors gracefully



@app.route('/model_performance', methods=['GET'])
def model_performance():
    url = 'https://drive.google.com/file/d/1kqnB4J8FuF1k8xLIvfbPE1jsqUwd3wVH/view?usp=sharing'
    model_handler = ModelHandler(url)
    performance = model_handler.evaluate_models()
    return jsonify({"RMSE": performance})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
