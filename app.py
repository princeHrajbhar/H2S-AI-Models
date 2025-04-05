from flask import Flask, render_template, request, jsonify
from src.predict import predict_yield

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        input_data = {
            'Temperature': float(request.form['temperature']),
            'Rainfall': float(request.form['rainfall']),
            'Soil_Type': request.form['soil_type'],
            'Fertilizer_Amount': float(request.form['fertilizer']),
            'Pesticide_Use': float(request.form['pesticide']),
            'Crop_Type': request.form['crop_type'],
            'Growing_Days': int(request.form['growing_days'])
        }

        # Make prediction
        prediction = predict_yield(input_data)

        return render_template('index.html',
                               prediction_text=f'Predicted Yield: {prediction:.2f} kg/ha')
    except Exception as e:
        return render_template('index.html',
                               prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)