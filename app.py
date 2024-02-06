import os 
from flask import Flask,render_template,request
import pickle
from PIL import Image
import tensorflow as tf
from joblib import load
import numpy as np
app=Flask(__name__)

def predict(values,dic):
    #DIABETES
    if len(values)==8:
        dic2 = {'NewBMI_Obesity 1': 0, 'NewBMI_Obesity 2': 0, 'NewBMI_Obesity 3': 0, 'NewBMI_Overweight': 0,
                'NewBMI_Underweight': 0, 'NewInsulinScore_Normal': 0, 'NewGlucose_Low': 0,
                'NewGlucose_Normal': 0, 'NewGlucose_Overweight': 0, 'NewGlucose_Secret': 0}
    
        if dic['BMI'] <= 18.5:
            dic2['NewBMI_Underweight'] = 1
        elif 18.5 < dic['BMI'] <= 24.9:
            pass
        elif 24.9 < dic['BMI'] <= 29.9:
            dic2['NewBMI_Overweight'] = 1
        elif 29.9 < dic['BMI'] <= 34.9:
            dic2['NewBMI_Obesity 1'] = 1
        elif 34.9 < dic['BMI'] <= 39.9:
            dic2['NewBMI_Obesity 2'] = 1
        elif dic['BMI'] > 39.9:
            dic2['NewBMI_Obesity 3'] = 1

        if 16 <= dic['Insulin'] <= 166:
            dic2['NewInsulinScore_Normal'] = 1

        if dic['Glucose'] <= 70:
            dic2['NewGlucose_Low'] = 1
        elif 70 < dic['Glucose'] <= 99:
            dic2['NewGlucose_Normal'] = 1
        elif 99 < dic['Glucose'] <= 126:
            dic2['NewGlucose_Overweight'] = 1
        elif dic['Glucose'] > 126:
            dic2['NewGlucose_Secret'] = 1

        dic.update(dic2)
        values2 = list(map(float, list(dic.values())))


        try:
            model=pickle.load(open('models\diabetes.pkl','rb'))
        except FileNotFoundError:
            print("Error: Mode file not found")
        except Exception as e:
            print(f"Error: failed to load the model. {e}")

        values=np.asarray(values2)
        return model.predict(values.reshape(1,-1))[0]
    elif len(values)==22:
        #breast cancer
        try:
            model = pickle.load(open('models\breast_cancer.pkl', 'rb'))
        except FileNotFoundError:
            print("Error: Model file not found.")
        except Exception as e:
            print(f"Error: Failed to load the model. {e}")

        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values)==13:
        from joblib import load
        try:
            model=load('models\heart.joblib')
        except FileNotFoundError:
            print("Error: Model file not found")
        except Exception as e:
            print(f"Error: failed to load the model. {e}")
        values=np.asarray(values)
        return model.predict(values.reshape(1,-1))[0]
    
    elif len(values)==24:
        from joblib import load
        try:
            model=load('models\kidney.joblib')
        except FileNotFoundError:
            print("Error: model file not found")
        except Exception as e:
            print(f"Error: FAILED TO LOAD THE MODEL. {e}")
        values=np.asarray(values) 
        return model.predict(values.reshape(1,-1))[0]

    elif len(values)==10:
        try:
            model=pickle.load(open('models\liver.pkl','rb'))
        except FileNotFoundError:
            print("Error: Model file not found") 
        except Exception as e:
            print(f"Error: failed to load the model {e}")

        values=np.asarray(values)
        print(values)
        return model.predict(values.reshape(1,-1))[0] 
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes",methods=['GET','POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer",methods=['GET','POST'])
def cancerPage():
    return render_template("breast_cancer.html")

@app.route("/heart",methods=['GET','POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney",methods=['GET','POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver",methods=['GET','POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria",methods=['GET','POST'])
def malariaPage():
    return render_template('malaria.html')  
@app.route("/pneumonia",methods=['GET','POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route('/predict',methods=['GET','POST'])
def predictPage():
    try:
        if request.method=='POST':
            to_predict_dict=request.form.to_dict();
            print(to_predict_dict)
            for key,value in to_predict_dict.items():
                try:
                    to_predict_dict[key]=int(value)
                except ValueError:
                    to_predict_dict[key]=float(value)
            
            to_predict_list=list(map(float,list(to_predict_dict.values())))
            print(to_predict_list)
            pred=predict(to_predict_list,to_predict_dict)
            return render_template('predict.html',pred=pred)
        
    except:
        message='PLEASE ENTER VALID DATA'
        return render_template('predict.html',message=message)
    return render_template('predict.html',pred=None)

@app.route("/malariapredict",methods=['POST','GET'])
def malariapredictPage():
    if request.method=='POST':
        try:
            img=Image.open(request.files['image'])
            img.save("uploads/image.jpg")
            img_path=os.path.join(os.path.dirname(__file__),'uploads/image.jpg')
            os.path.isfile(img_path)
            img=tf.keras.utils.load_img(img_path,target_size=(128,128))
            img=np.expand_dims(img,axis=0)
            model=tf.keras.models.load_model("models/malaria.h5")
            pred=np.argmax(model.predict(img))
            return render_template('malaria_predict.html',pred=pred)
        
        except:
            message="Please upload na image"
            return render_template('malaria.html',message=message)
    return render_template('malaria_predict.html',pred=None)
@app.route("/pneumoniapredict",methods=['POST','GET'])
def pneumoniapredictPage():
    if request.method=='POST':
        try:
            img=Image.open(request.files['image']).convert('L')
            img.save("uploads/image.jpg")
            img_path=os.path.join(os.path.dirname(__file__),'uploads/image.jpg')
            os.path.isfile(img_path)
            img=tf.keras.utils.load_img(img_path,target_size=(128,128))
            img=tf.keras.utils.img_to_array(img)
            img=np.expand_dims(img,axis=0)
            model=tf.keras.models.load_model("models/pneumonia.h5")
            pred=np.argmax(model.predict(img))

            return render_template('pneumonia_predict.html',pred=pred)
        except:
            message="Please upload an image"
            return render_template('pneumonia.html',message=message)
    return render_template('pnemonia_predict.html',pred=None)


if __name__=='__main__':
    app.run(debug=True)