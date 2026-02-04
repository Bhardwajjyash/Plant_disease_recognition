import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_extras.star_rating import star_rating


def model_prediction(test_image):
        model = tf.keras.models.load_model("trained_model.keras")
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr  = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])#convert single image to a batch
        print(input_arr.shape)
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        return result_index
    
    
#sidebar
st.sidebar.title("Dashbord")
app_mode = st.sidebar.selectbox("select page",["Home","About","Disease Recognition",])
# Home page
if(app_mode=="Home"):
        st.header("PLANT RECOGNITION SYSTEM")
        image_path = "home_page.png"
        st.image(image_path,width="content")
        st.markdown("""
## Plant Disease Recognition System

This platform is designed to assist in the identification of plant diseases through image-based analysis. By using modern machine learning techniques, the system helps detect plant health issues at an early stage, enabling timely and informed decision-making.

The objective of this project is to support better crop management by providing a fast, reliable, and easy-to-use disease recognition solution.

---

### How It Works
1. Upload an image of a plant leaf through the Disease Recognition section.
2. The system analyzes the image using a trained deep learning model.
3. A prediction is generated to indicate the possible disease affecting the plant.

---

### Key Features
- Accurate disease identification using machine learning models  
- Simple and intuitive user interface  
- Fast processing with instant results  

---

### Getting Started
To begin, open the Disease Recognition page from the sidebar and upload a clear image of the plant leaf you want to analyze.

---

### About the Project
Additional information about the project, including its purpose, methodology, and development details, can be found in the About section.
""")
elif(app_mode=="About"):
        st.header("About")
        st.markdown("""
#### About the Dataset

The dataset used in this project has been created through offline data augmentation based on an original publicly available dataset. The original source dataset is hosted in a GitHub repository and serves as the foundation for this work.

It contains approximately 87,000 RGB images of crop leaves, including both healthy and diseased samples. The images are classified into 38 distinct categories, each representing a specific crop–disease combination.

To ensure proper model training and evaluation, the dataset is divided into training and validation sets using an 80:20 split while maintaining the original directory structure. Additionally, a separate test directory was created at a later stage to evaluate model predictions on unseen data.

#### Dataset Structure
1. **Training Set** – 70,295 images  
2. **Validation Set** – 17,572 images  
3. **Test Set** – 33 images  
""")
#prediction page
elif(app_mode=="Disease Recognition"):
        
        st.header("Disease Recognition")
        test_image = st.file_uploader("Choose an Image:")
        if(st.button("Show Image")):
                st.image(test_image,width="content")
        #Actual prediction
        if(st.button("predict")):
                with st.spinner("please wait..."):
                        
                        st.write("Our prediction")
                        result_index = model_prediction(test_image)
                        #class name
                        class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
                        st.success("Model prediction : {}".format(class_name[result_index]))
                        st.text("Rating of the Match")
                        star_rating(5)
                        st.balloons()