from flask import Flask, request, render_template
import pickle
import pandas as pd

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load model yang sudah dilatih
model = pickle.load(open('saved_model.pkl/best_rf_model.pkl', 'rb'))

# Route utama untuk form input
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form HTML
    glucose = request.form['Glucose']
    bmi = request.form['BMI']
    age = request.form['Age']

    # Buat dataframe dari input user
    new_data = pd.DataFrame({'Glucose': [float(glucose)], 'BMI': [float(bmi)], 'Age': [float(age)]})

    # Lakukan prediksi menggunakan model yang telah dilatih
    prediction = model.predict(new_data)

    # Hasil prediksi
    if prediction[0] == 1:
        result = "Anda berpotensi terkena diabetes."
    else:
        result = "Anda berkemungkinan tidak terkena diabetes."

    # Kirimkan kembali nilai input dan hasil prediksi
    return render_template('index.html', prediction_text=result, glucose=glucose, bmi=bmi, age=age)

if __name__ == "__main__":
    app.run(debug=True)
