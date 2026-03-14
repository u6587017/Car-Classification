# Car-Classification

This project practices on transfer-learning with ResNet50V2 model to classify car from 20 brands (3235 images) 
A high-performance computer vision API built with FastAPI and TensorFlow. This service accepts car images and returns the predicted car type with a confidence score.
It is containerized using Docker, the image was pushed to store on AWS ECR and deployed as a serverless microservice on AWS App Runner for automatic scaling and high availability.

## Tech Stack
Backend: FastAPI (Python 3.11)
ML Engine: TensorFlow / Keras
Image Processing: OpenCV
Deployment: Docker, Amazon ECR, AWS App Runner

## API Usage
Endpoint: POST /classify
Uploads an image file for classification.

Request:
Method: POST
Content-Type: multipart/form-data
Body: file (Binary Image)
Success Response:
~~~
{
  "filename": "car_photo.jpg",
  "prediction": "Nissan",
  "confidence": 0.9845
}
~~~
## Dataset example:
<img width="1611" height="403" alt="input_example" src="https://github.com/user-attachments/assets/f3974e14-4bb1-4fbb-aa79-85a3008da51d" />
<br />

Parameters:
Optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'], epochs=8

## Accuracy and Loss
![acc](acc.png)
<br />
![loss](Loss.png)
## Evaluation
Loss: 1.23, Accuracy: 0.657
