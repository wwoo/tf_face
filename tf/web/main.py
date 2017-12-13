from flask import Flask, Response, request, jsonify, render_template
from PIL import Image, ImageDraw
from io import BytesIO
from googleapiclient import discovery
from oauth2client.service_account import ServiceAccountCredentials

import numpy
import base64
import cStringIO
import os
import json
import logging

import resources, classes

PROJECT = 'your-project-id'
MODEL_NAME = 'face_test'
IMAGE_SIZE = 96

SCOPES = 'https://www.googleapis.com/auth/cloud-platform'
MAX_PREDICTIONS = 5

def get_vision_service():
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        os.path.join(resources.__path__[0], 'vapi-acct.json'), SCOPES)
    return discovery.build('vision', 'v1', credentials=credentials)

def get_cloudml_service():
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        os.path.join(resources.__path__[0], 'vapi-acct.json'), SCOPES)
    return discovery.build('ml', 'v1', credentials=credentials)

flask_app = Flask(__name__)

@flask_app.route('/', methods=['GET'])
def get_index():
    return render_template("index.html",
        results = 'false',
        error = 'false')

@flask_app.route('/', methods=['POST'])
def classify_file():
    # Detect where the face is in the picture
    f = request.files['file']
    image_content = f.read()
    top, left, bottom, right, img_str = detect_face(image_content)

    if img_str is not None:
        # Get prediction from model server
        pred = recognise_face(img_str)

        # Find the top classes and scores
        top_pred = pred.argsort()[-MAX_PREDICTIONS:][::-1].tolist()
        top_scores = numpy.sort(pred)[-MAX_PREDICTIONS:][::-1]
        top_scores = numpy.multiply(top_scores, 100)

        # Draw a rectangle to identify the face
        im = Image.open(BytesIO(image_content))
        dr = ImageDraw.Draw(im)
        dr.rectangle((left, top, right, bottom), outline="green")
        buffer = cStringIO.StringIO()
        im.save(buffer, format="JPEG")

        return render_template('index.html',
            results = 'true',
            error = 'false',
            image_b64 = base64.b64encode(buffer.getvalue()),
            prediction_1 = classes.CLASSES[top_pred[0]],
            prediction_2 = classes.CLASSES[top_pred[1]],
            prediction_3 = classes.CLASSES[top_pred[2]],
            prediction_4 = classes.CLASSES[top_pred[3]],
            prediction_5 = classes.CLASSES[top_pred[4]],
            score_1 = top_scores[0],
            score_2 = top_scores[1],
            score_3 = top_scores[2],
            score_4 = top_scores[3],
            score_5 = top_scores[4])

    return render_template('index.html',
        results = 'false',
        error = 'true')

def detect_face(image_content):
    request_dict = [{
        'image': {
            'content': base64.b64encode(image_content)
            },
        'features': [{
            'type': 'FACE_DETECTION',
            'maxResults': 1,
            }]
        }]

    (top, left, bottom, right, img_str) = 0, 0, 0, 0, None

    try:
        vision_svc = get_vision_service()
        request = vision_svc.images().annotate(body={
            'requests': request_dict
        })
        response = request.execute()

        face_bounds = response['responses'][0]['faceAnnotations'][0]['fdBoundingPoly']['vertices']

        if (len(face_bounds) == 4):
            left = face_bounds[0]['x']
            top = face_bounds[0]['y']
            right = face_bounds[2]['x']
            bottom = face_bounds[2]['y']

            # Crop the image to just the face only`
            im = Image.open(BytesIO(image_content))
            face_im = im.crop((left, top, right, bottom)).resize((IMAGE_SIZE, IMAGE_SIZE))

            # Convert face image to base64
            image_buffer = cStringIO.StringIO()
            face_im.save(image_buffer, format="JPEG")
            img_str = base64.b64encode(image_buffer.getvalue())

    except Exception, e:
        logging.exception("Something went wrong [detect_face]")

    return top, left, bottom, right, img_str

def recognise_face(img_str):
    parent = 'projects/{}/models/{}'.format(PROJECT, MODEL_NAME)
    pred = None
    request_dict = {
        "instances": [
            {
                "image_bytes": {
                    "b64": img_str
                }
            }
        ]
    }

    try:
        cloudml_svc = get_cloudml_service()
        request = cloudml_svc.projects().predict(name=parent, body=request_dict)
        response = request.execute()
        pred = response['predictions'][0]['scores']
        pred = numpy.asarray(pred)

    except Exception, e:
        logging.exception("Something went wrong [recognise_face]")

    return pred

def main():
    flask_app.run(host='127.0.0.1', port=8080)

if __name__ == '__main__':
    main()
