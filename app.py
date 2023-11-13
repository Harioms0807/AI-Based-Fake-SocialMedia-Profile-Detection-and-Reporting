# app.py
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load the model architecture from 'model.json' and the weights from 'model.h5'
with open('model.json', 'r') as model_json:
  model = tf.keras.models.model_from_json(model_json.read())
model.load_weights('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    # Get input data from the form
    input_data = request.form

    # Extract features from the input data
    statuses_count = input_data['statuses_count']
    followers_count = input_data['followers_count']
    friends_count = input_data['friends_count']
    favourites_count = input_data['favourites_count']

    listed_count = input_data['listed_count']
    geo_enabled = input_data['geo_enabled']
    profile_use_background_image = input_data['profile_use_background_image']

    # Make predictions using your model
    input_features = [statuses_count, followers_count, friends_count, favourites_count, listed_count, geo_enabled, profile_use_background_image]
    final=[np.array(input_features)]
    prediction = model.predict([final])

    # Interpret the prediction result (e.g., "Fake" or "Legit")
    prediction_result = "Fake" if prediction[0] > 0.5 else "Legit"

    return jsonify({'prediction': prediction_result})
if __name__ == '__main__':
    app.run(debug=True)
