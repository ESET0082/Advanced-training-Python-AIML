### How the App Works Internally

##### step 1 : The user opens the website and sees a form to enter booking details.

##### step 2 :After filling in the details and submitting, the information is sent to the backend app.

##### step 3 :The app collects the input data and prepares it in a proper format.

##### step 4 :A saved preprocessor cleans and converts the data into numbers the model can understand.

##### step 5 :The trained Keras model is then loaded.

##### step 6 :The model makes a prediction based on the processed input.

##### step 7 :The prediction result is converted into a user-friendly message.

##### step 8 :That message is displayed on a new page showing the final result.

##### When deployed online, Docker automatically handles everything — training if needed and running the app continuously.




### Booking Prediction Web App

##### A simple Flask web application to enter booking details and predict outcomes using a saved Keras model (final_model.keras) and a scikit-learn preprocessor (preprocessor.pkl). This repository includes a Dockerfile so you can containerize the app and deploy it to Render (or another hosting provider).

##### Live demo: https://day-24-dl-project.onrender.com/


### Project overview

##### This project provides a simple web form (HTML) to collect booking information from a user. The Flask app (app.py) loads a pre-trained Keras model (final_model.keras) and a preprocessor (preprocessor.pkl) to transform inputs and produce predictions. index.html is the input form and result.html displays the model's output.

##### The included Dockerfile and helper scripts allow the container to check for the model at startup, train it if missing using Train_model.py, and then start the Flask app. This makes the container self-sufficient on first run.

### Repository structure

##### booking-prediction-app/
##### ├─ app.py # Flask app that serves index and result pages
##### ├─ Train_model.py # Training script to produce final_model.keras and preprocessor.pkl
##### ├─ final_model.keras # (optional) pre-trained model file
##### ├─ preprocessor.pkl # (optional) saved preprocessing pipeline
##### ├─ requirements.txt
##### ├─ Dockerfile
##### ├─ start.sh # entrypoint script used by Dockerfile
##### ├─ templates/
##### │ ├─ index.html
##### │ └─ result.html

### Requirements

##### scikit-learn
##### uvicorn
##### fastapi[standard]
##### pandas
##### numpy
##### python-multipart
##### joblib
##### jinja2
##### tensorflow