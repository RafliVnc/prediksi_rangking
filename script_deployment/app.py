import pickle
import numpy as np
import json
from flask import Flask, request, render_template
app = Flask(__name__)

#load data
with open("model/grade_siswa.pkl","rb") as model_file:
    loaded_model = pickle.load(model_file)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/tes")
def tes():
    args = request.args
    nama =args.get('nama', default="Udin")

    return {"status":"SUCCESS",
            "Message":f"Nama : {nama}"},200


@app.route("/predict/json")
def jason():
    args = request.args
    Medu = args.get('Medu',default=0.0,type=float)
    Fedu = args.get('Fedu',default=0.0,type=float)
    studytime = args.get('studytime',default=0.0,type=float)
    failures = args.get('failures',default=0.0,type=float)
    absences=  args.get('absences',default=0.0,type=float)
    G1 = args.get('G1',default=10.0,type=float)
    G2 = args.get('G2',default=20.0,type=float)
    higher_yes = args.get('higher_yes',default=1.0,type=float)
    new_data =[[Medu,Fedu,studytime,failures,absences,G1,G2,higher_yes]]
    res_predict=loaded_model.predict(new_data)
    res = round(res_predict[0])

    return {"status":"SUCCESS",
            "input":{
                "medu":Medu,
                "fedu":Fedu,
                "Studytime":studytime,
                "failures":failures,
                "absencs": absences,
                "G1":G1,
                "G2":G2,
                "Higer":higher_yes,
            },
            "result":res},200  


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    res_predict = loaded_model.predict(features)
    res = round(res_predict[0])
    return render_template("hasil.html", prediction_text = "{}".format(res))


# link test
# http://127.0.0.1:5000/predict?Medu=3&Fedu=3&studytime=2&failures=0&absences=0&G1=13&G2=14&higher_yes=1

if __name__ == "__main__":
    app.run(debug=True)