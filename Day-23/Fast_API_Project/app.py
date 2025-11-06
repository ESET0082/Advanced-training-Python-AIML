# from typing import Union

# from fastapi import FastAPI, Form
# from fastapi.templating import Jinja2Templates
# from fastapi import Request

# from pydantic import BaseModel
# import joblib
# from sklearn.datasets import load_iris
# import pandas as pd
# import numpy as np

# app = FastAPI()

# model = joblib.load('iris_model.pkl')
# templates = Jinja2Templates(directory="templates")

# class IrisInput(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float

# class IrisPrediction(BaseModel):
#     predicted_class: int
#     predicted_class_name: str

# @app.get("/")
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": {"request": request}})


# @app.post("/predict/",response_model=IrisPrediction)
# def predict(
#     request: Request,
#     sepal_length: float = Form(...),
#     sepal_width: float = Form(...),
#     petal_length: float = Form(...),
#     petal_width: float = Form(...)
# ):


# # def predict(data:IrisInput):
    
#     ## Prepare input data
#     InputData = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
#                              columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    
#     ## Model Prediction
#     predict_class = model.predict(InputData)[0]
#     predict_class_name = load_iris().target_names[predict_class]
#     # return IrisPrediction(predicted_class=predict_class, predicted_class_name=predict_class_name)
#     return templates.TemplateResponse("result.html", {"request": {"request": request},
#                                                      "predicted_class": int(predict_class),
#                                                      "predicted_class_name": str(predict_class_name),
#                                                      "sepal_length": sepal_length,
#                                                      "sepal_width": sepal_width,
#                                                      "petal_length": petal_length,
#                                                      "petal_width": petal_width
#                                                      })




# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8088)

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.datasets import load_iris
 
# Load the trained model
model = joblib.load("iris_model.pkl")
 
app = FastAPI()
 
# Set up templates directory
templates = Jinja2Templates(directory="template")
 
 
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
 
class IrisPrediction(BaseModel):
    predicted_class: int
    predicted_class_name: str
 
 
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
   
 
 
@app.post("/predict", response_model=IrisPrediction)
def predict(request: Request,
            sepal_length: float = Form(...),
            sepal_width: float = Form(...),
            petal_length: float = Form(...),
            petal_width: float = Form(...)):
    # prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
 
    # make prediction
    predicted_class = model.predict(input_data)[0]
    predicted_class_name = load_iris().target_names[predicted_class]
    return templates.TemplateResponse(
        "result.html",
        {"request": request,
         "predicted_class": predicted_class,
         "predicted_class_name": predicted_class_name,
         "sepal_length": sepal_length,
         "sepal_width": sepal_width,
         "petal_length": petal_length,
         "petal_width": petal_width})
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
 
