"""
Run a rest API exposing the yolov5s object detection model
"""
import io
import pickle
import argparse

import torch
from flask import Flask, request

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5s"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_bytes = request.files["image"]
        image_bytes = io.BytesIO(image_bytes.read())
        img = pickle.load(image_bytes)
       
        results = model(img, size=256, conf=0.01, iou=0.01)
        return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = torch.hub.load("alexjercan/yolov5", "yolov5s", force_reload=True)  # force_reload to recache
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
