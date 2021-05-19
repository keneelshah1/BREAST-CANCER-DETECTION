from flask import render_template, Flask
from flask_bootstrap import Bootstrap
from PIL import Image
import pickle
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

# Flask app gets defined
app = Flask(__name__)

#---------------------- TISSUE MODEL -------------------------------#
# model = load_model(r'models\model_cancer_detector.h5')
model_tissue=load_model(r'C:\Users\admin\Desktop\pythonProject\FN_BCD\models\IV3.h5')

#---------------------- Mammography MODEL -------------------------------#
model_BMN=load_model(r'C:\Users\admin\Desktop\pythonProject\FN_BCD\models\MAMO_BCD_BMN.h5')
model_FDG=load_model(r'C:\Users\admin\Desktop\pythonProject\FN_BCD\models\MAMO_FDG_ResNet50.h5')

#---------------------- Numeric MODEL -------------------------------#
model_numeric = pickle.load(open('C:\\Users\\admin\\Desktop\\pythonProject\\FN_BCD\\models\\SVC_lin.sav', 'rb'))

upload_files = 'C:\\Users\\admin\\Desktop\\pythonProject\\FN_BCD\\static'

#---------------------- TISSUE MODEL Function -------------------------------#
def model_predict_Tissue(img_path, model_tissue) :
    test_image = image.load_img(img_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = test_image/255
    test_image = np.expand_dims(test_image, axis=0)
    result = model_tissue.predict(test_image)
    r = np.max(result)
    per = r * 100
    percentage = round(per, 4)
    categories = ['IDC NEGATIVE', 'IDC POSITIVE']

    # process your result for human
    pred_class = result.argmax()
    output = categories[pred_class]
    return percentage,output

#---------------------- Mammography MODEL Finction-------------------------------#
def model_predict_mammo(img_path, model_BMN,model_FDG) :
    test_image = img_path
    image_result = Image.open(test_image)
    test_image = image.load_img(test_image, target_size=(224, 224, 3))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    result_BMN = model_BMN.predict(test_image)
    result_FDG = model_FDG.predict(test_image)

    r_BMN = np.max(result_BMN)
    r_FDG = np.max(result_FDG)

    per_BMN = r_BMN * 100
    per_FDG = r_FDG * 100

    percentage_BMN = round(per_BMN, 4)
    percentage_FDG = round(per_FDG, 4)

    categories_BMN = ['BENIGN IMAGE', 'MALIGNANT IMAGE', 'NORMAL IMAGE']
    categories_FDG = ['Fatty Tissue --> ', 'Dense-glandular tissue --> ', 'Fatty-glandular Tissue --> ']

    # process your result for human
    image_result = plt.imshow(image_result)

    output_BMN = categories_BMN[np.argmax(result_BMN)]
    output_FDG = categories_FDG[np.argmax(result_FDG)]

    o = output_FDG + output_BMN

    return o,output_FDG,output_BMN,per_BMN,per_FDG,image_result

#---------------------- Call Main Page -------------------------------#
@app.route('/')
def index():
    return render_template('index.html')

#---------------------- Patient information -------------------------------#
@app.route('/info',methods = ["GET","POST"])
def info():
    if request.method == "POST":
        info.Name = request.form['Name']
        info.Age = request.form['Age']
        info.Id = request.form['Id']
        seperator = '_'
        info.concat = info.Name + seperator + info.Id
        return render_template('selector.html')

#---------------------- Calls the Tissue based Html Page -------------------------------#
@app.route('/BCD_Tissue')
def BCD_Tissue():
    return render_template('BCD_Tissue.html')
#---------------------- Function for Prediction on Tissue based Model -------------------------------#
@app.route('/predict_tissue',methods = ["GET","POST"])
def upload_predict_tissue():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(upload_files,image_file.filename)
            image_file.save(image_location)
            pred = model_predict_Tissue(image_location,model_tissue)
            return render_template('result_tissue.html', output=pred, image_loc=image_file.filename, Name=info.Name, Age=info.Age, Id=info.Id, concat=info.concat)

#---------------------- Calls the Mammography based Html Page -------------------------------#
@app.route('/BCD_Mammo')
def BCD_Mammo():
    return render_template('BCD_Mammo.html')
#---------------------- Function for Prediction on Mammography based Model -------------------------------#
@app.route('/predict_mammo',methods = ["GET","POST"])
def upload_predict_mammo():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(upload_files,image_file.filename)
            image_file.save(image_location)
            pred = model_predict_mammo(image_location,model_BMN,model_FDG)
            return render_template('result_mammo.html', output=pred,image_loc=image_file.filename, Name=info.Name, Age=info.Age, Id=info.Id, concat=info.concat)

#---------------------- Calls the Numeric based Html Page -------------------------------#
@app.route('/BCD_Numeric')
def BCD_Numeric():
    return  render_template('BCD_Numeric.html')
#---------------------- Function for Prediction on Numeric based Model -------------------------------#
@app.route('/predict_numeric', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                     'mean smoothness', 'mean compactness', 'mean concavity',
                     'mean concave points', 'mean symmetry', 'mean fractal dimension',
                     'radius error', 'texture error', 'perimeter error', 'area error',
                     'smoothness error', 'compactness error', 'concavity error',
                     'concave points error', 'symmetry error', 'fractal dimension error',
                     'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                     'worst smoothness', 'worst compactness', 'worst concavity',
                     'worst concave points', 'worst symmetry', 'worst fractal dimension']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model_numeric.predict(df)

    if output == 1:
        res_val = "** breast cancer **"
    else:
        res_val = "no breast cancer"

    return render_template('result_numeric.html', prediction_text=res_val, Name=info.Name, Age=info.Age, Id=info.Id, concat=info.concat)#Name=Name,Age=Age,Id=Id

#---------------------- Function for returning to main Index Page -------------------------------#
@app.route('/vmd_timestamp')
def vmd_timestamp():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)