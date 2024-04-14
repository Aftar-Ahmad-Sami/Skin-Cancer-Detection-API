# This module sets up a Flask-based web service with CORS enabled that provides an API 
# for simple addition and image prediction for medical images. The image prediction 
# leverages deep learning models from the MONAI framework, image preprocessing via OpenCV, 
# and PyTorch for model inference.

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import cv2
import os
from monai.inferers import SimpleInferer, SlidingWindowInferer, SaliencyInferer
from monai.data import DataLoader
from monai.networks.nets import DenseNet121
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    LoadImage,
    ScaleIntensity,
    ToTensor,
)
import base64


app = Flask(__name__)
CORS(app)

# A list of possible predictions classifications
label_list = ['Benign', 'Malignant'] 
height = 224  # The height of the images to be used for the model
width = 224   # The width of the images to be used for the model
class_map = {}
num_class = len(label_list)
for i in range(num_class):
    class_map[i] = label_list[i]

# Defines transformation to apply to test images before sending them to the model
test_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

class MedNISTDataset(torch.utils.data.Dataset):
    """
    A dataset class that holds medical images along with their corresponding labels and applies
    the given transformations on the images.
    
    Attributes:
        image_files: A list of file paths or array of images to be loaded.
        labels: A list of associated labels with the images.
        transforms: A callable that applies transformations to the data.

    Methods:
        __len__: Returns the number of items in the dataset.
        __getitem__: Returns the transformed image and its label at the specified index.
    """
    
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Uncomment the following lines to load the trained model
# model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2).to(device)
# model.load_state_dict(torch.load("/path/to/your/model.pth", map_location=device))
# model = torch.jit.load('/path/to/your/traced/model.pt')
# model.eval()

def d2Kmeans(img, k):
    """
    Apply 2-dimensional KMeans algorithm on an image.

    Parameters:
        img: An input image array.
        k: The number of clusters to form as well as the number of centroids to generate.

    Returns:
        A two-dimensional array where each pixel is labeled with a cluster index.
    """
    Z = img.reshape((-1, 1))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return label.reshape(img.shape)


# The following function definitions provide an image segmentation pipeline
# which includes converting an image to grayscale, clustering, selecting the relevant cluster,
# and creating round boundaries around significant regions in the image.
# This pre-processing step could be useful in highlighting certain features within medical images
# and omitting unnecessary parts of the image before sending it for model prediction.

@app.route('/add', methods=['POST'])
def add_numbers():
    """
    An API endpoint for adding two numbers sent as JSON in the request body.
    
    Returns:
        A JSON object containing the sum of the numbers, or an error message
        if the request does not contain the necessary information.
    """
    try:
        data = request.get_json()
        num1 = data['num1']
        num2 = data['num2']
        result = num1 + num2
        return jsonify({'sum': result})
    except KeyError:
        return jsonify({'error': 'Please provide both num1 and num2 as JSON in the request body.'}), 400


@app.route('/predict', methods=['POST'])
def predict():
    """
    An API endpoint to predict whether an uploaded medical image is classified as 'Benign' or 
    'Malignant' through segmentation and Deep Learning model inference. 
    
    The uploaded image is preprocessed (resized, segmented), and then a prediction is made using 
    a pre-loaded deep learning model. The result and the processed image are returned as a JSON object.
    
    Returns:
        A JSON response containing the class prediction and the processed image in base64 encoding, 
        or an error message with details if an exception is encountered.
    """
    try:
        image_file = request.files['image']
        image_path = '/path/to/save/temp_image.jpg'
        image_file.save(image_path)
        image = cv2.imread(image_path)
        resized_img = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(image_path, resized_img)
        img = segment_image(image)
        cv2.imwrite('/path/to/save/processed_img.jpg', img)
        
        test_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])
        model = torch.jit.load('/path/to/load/your/model/Skin_Cancer_Detection-20.pt')
        model.eval()
        
        ixr = test_transforms(image_path).to(device).unsqueeze(0)
        ixr = ixr.permute(0,3,1,2)
        lis = [image_path]
        test_n_ds = MedNISTDataset(lis, [-1], test_transforms)
        test_n_loader = DataLoader(test_n_ds, batch_size=64, num_workers=2)

        with torch.no_grad():
            for test_data in test_n_loader:
                test_images, test_labels = test_data[0], test_data[1]
                pred = model(test_images).argmax(dim=1)
        
        with open('/path/to/save/processed_img.jpg', "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({'result': class_map[pred[0].item()], 'image_data': encoded_image})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"result-error": str(e)})


@app.route('/')
def hello():
    """
    A simple API endpoint for testing that the service is running.
    
    Returns:
        A plain text greeting.
    """
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)  # Use a port number that is not in use

# Note: Additional inline comments inside specific functions and code blocks could further clarify complex logic. 
# Some code blocks are commented out and require activation by providing proper file paths and ensuring the 
# availability of trained model files.
