from flask import Flask,request, render_template, url_for
import pickle
import pandas as pd

app = Flask(__name__)

model=pickle.load(open('ridge_model.pkl','rb'))
model2=pickle.load(open('DT_model.pkl','rb'))

@app.route('/')
def home():
    bg_image_url = url_for('static', filename='bg3.jpg')
    return render_template("index.html",bg_image_url=bg_image_url)

@app.route('/premium')
def pre():
    bg_image_url = url_for('static', filename='bg3.jpg')
    return render_template("premium.html",bg_image_url=bg_image_url)

@app.route('/predict',methods=['POST','GET'])
def predict_basic():
    bg_image_url = url_for('static', filename='bg3.jpg')
    int_features=[float(x) for x in request.form.values()]
    feature_names = ['temp_avg', 'precip','humidity']
    predictor=["precip","temp_avg","humidity"]
    new_data_df = pd.DataFrame([int_features], columns=feature_names)
    
    prediction = model2.predict(new_data_df[predictor])
    print(int_features)
    print(new_data_df)
    # prediction=model.predict_proba(final)
    print(f"prediction={prediction}")
    prediction[0]=float("{:.2f}".format(prediction[0]))
    return render_template('index.html',bg_image_url=bg_image_url,pred=f'Next day the temperature will be: {prediction[0]} degree celsius')


@app.route('/predict_pre',methods=['POST','GET'])
def predict_pre():
    bg_image_url = url_for('static', filename='bg3.jpg')
    int_features=[float(x) for x in request.form.values()]
    feature_names = ['temp_avg', 'precip','humidity']
    predictor=["precip","temp_avg","humidity"]
    new_data_df = pd.DataFrame([int_features], columns=feature_names)
    
    prediction = model.predict(new_data_df[predictor])
    print(int_features)
    print(new_data_df)
    # prediction=model.predict_proba(final)
    print(f"prediction={prediction}")
    prediction[0]=float("{:.2f}".format(prediction[0]))
    return render_template('premium.html',bg_image_url=bg_image_url,pred=f'Next day the temperature will be: {prediction[0]} degree celsius')


if __name__ == '__main__':
    app.run(debug=True)
