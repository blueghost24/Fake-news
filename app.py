from flask import Flask, render_template, request
import joblib
from utils.preprocessing import preprocess_text

app = Flask(__name__)

# Load model and vectorizer
vectorizer, model = joblib.load('model/fake_news_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    if request.method == 'POST':
        title = request.form['title']
        text = preprocess_text(f"{title}")
        text_vector = vectorizer.transform([text])
        result = model.predict(text_vector)[0]
        prediction = 'Fake' if result == 1 else 'Real'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
