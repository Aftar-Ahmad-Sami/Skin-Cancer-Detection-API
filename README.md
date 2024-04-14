# Skin Cancer Detection API Documentation

## Introduction

This documentation provides information about a Flask-based web service that exposes an API for making predictions on medical images using deep learning. The service uses the MONAI framework for inference, incorporating OpenCV for image processing, and leveraging PyTorch for the execution of the model. The API supports the classification of images into categories such as 'Benign' and 'Malignant.'

## Setup and Installation

To utilize this web service, you need to set up a Python environment and install the required packages. Ensure you have Python 3 installed, then follow these steps:

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```
   
2. Activate the virtual environment:

   - On Windows:
     ```
     .\venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```
     source venv/bin/activate
     ```

3. Install the necessary Python packages:
   ```
   pip install flask flask-cors numpy torch opencv-python monai
   ```

4. Configure the model file paths:
   - Uncomment the lines pertaining to model loading in the code, and replace `"/path/to/your/model.pth"` and `"/path/to/your/traced/model.pt"` with the correct paths to your pretrained model files.

5. Set the image file paths:
   - Modify `image_path` and the paths to save the processed images as necessary within the `predict` endpoint to specify where the uploaded and processed images should be stored.

## Usage Guide

To run the web service, execute the script from the command line:

```
python your_script_name.py
```

Once running, the web service will expose the following endpoints:

- `/predict`: Accepts a POST request with an image file and returns a prediction of 'Benign' or 'Malignant' along with the processed image encoded in base64.
- `/`: A simple GET endpoint that returns a 'Hello, World!' message for testing the service availability.

### Making Predictions

To make a prediction, perform a POST request to the `/predict` endpoint with a form-data payload including an image file under the key 'image'. Ensure your web service script is running before sending the request. An example using `curl` is as follows:

```shell
curl -X POST -F "image=@path/to/your/medical/image.jpg" http://localhost:5000/predict
```

Replace `path/to/your/medical/image.jpg` with the path to the medical image you want to analyze.

## Prerequisites or Dependencies

This code has several prerequisites:

- Flask (Web framework)
- Flask-CORS (Cross-origin resource sharing for Flask)
- NumPy (Array computations)
- Torch (PyTorch deep learning framework)
- OpenCV (Image processing)
- MONAI (Medical Open Network for AI)

These can all be installed via `pip` as shown in the Setup Instructions.

## Contributing

To contribute to this project, please follow the steps below:

1. Clone the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

Ensure that you include unit tests for any new features or bug fixes and that your code complies with the existing style.

## License

This project is licensed under the Apache License, Version 2.0 (January 2004). You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

For full details pertaining to the license, refer to the LICENSE file included with the code.

---

All users and contributors are expected to comply with the terms and conditions outlined in the Apache License 2.0 when using or contributing to this project.
