# Plant_disease_recognition
PLANT DISEASE RECOGNITION SYSTEM
===============================

Project Overview
----------------
The Plant Disease Recognition System is a machine learning–based application designed to identify diseases in crop leaves using image analysis. The system helps in early detection of plant diseases, enabling timely intervention and better crop management.

The application uses a trained deep learning model to analyze uploaded images of plant leaves and predict the corresponding disease class. A user-friendly web interface is provided using Streamlit for easy interaction.

Live Application:
https://plant-disease-recognition-by-bhardwajjyash.streamlit.app/


Features
--------
- Image-based plant disease detection
- Supports multiple crop and disease categories
- Simple and intuitive web interface
- Fast prediction using a trained deep learning model
- Separate training, validation, and testing datasets


Technology Stack
----------------
- Programming Language: Python
- Machine Learning Framework: TensorFlow / Keras
- Web Framework: Streamlit
- Image Processing: PIL, NumPy
- Model Type: Convolutional Neural Network (CNN)


Dataset Description
-------------------
The dataset used in this project is generated through offline data augmentation from an original publicly available dataset.

- Approximately 87,000 RGB images
- 38 different classes (healthy and diseased crop leaves)
- Dataset split:
  - Training Set: 70,295 images
  - Validation Set: 17,572 images
  - Test Set: 33 images

The directory structure is preserved to ensure compatibility with deep learning training workflows.


Project Structure
-----------------
plant_disease/
│
├── main.py                 # Streamlit application entry point
├── model/                  # Trained ML model files
├── dataset/                # Training, validation, and test data
├── requirements.txt        # Project dependencies
└── README.txt              # Project documentation


How It Works
------------
1. The user uploads an image of a plant leaf through the web interface.
2. The image is preprocessed and passed to the trained deep learning model.
3. The model predicts the most likely disease category.
4. The prediction result is displayed on the interface.


Installation and Setup
----------------------
1. Clone the repository:
   git clone <repository-url>

2. Create and activate a virtual environment (optional but recommended).

3. Install dependencies:
   pip install -r requirements.txt

4. Run the Streamlit application:
   streamlit run main.py


Use Cases
---------
- Assisting farmers in early disease detection
- Educational demonstrations of applied machine learning
- Academic projects related to agriculture and AI
- Research and experimentation with image classification models


Future Enhancements
-------------------
- Addition of more crop and disease classes
- Improved model accuracy with larger datasets
- Disease treatment and prevention recommendations
- Mobile-friendly deployment


Author
------
Developed by:
Yash Bhardwaj

This project is created for educational and practical demonstration purposes in the field of machine learning and agricultural technology.
