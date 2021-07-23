from flask import Flask, render_template, request
from model_structure import prediction


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST','GET'])
def predict():
   
    input_text = request.form['text']
    output_dict = prediction(input_text)
    # result = {'anger':12, 'love':45,'sadness':78, 'joy':86,'surprise':56,'fear':78}
    return render_template('index.html',result = output_dict, input_text = input_text)


if __name__ == '__main__':
    app.run(debug=False)