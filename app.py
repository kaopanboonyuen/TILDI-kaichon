import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
#import time
fig = plt.figure()

# # with open("custom.css") as f:
# #     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('TILDI KAICHON')

# st.markdown("<h1 style='text-align: center; color: black;'>TILDI KAICHON</h1>", unsafe_allow_html=True)


# #st.markdown("TILDI-Kaichon Classification")

# st.markdown("<h3 style='text-align: center; color: black;'>Thai Fighting Cock (Fighting Rooster) Classification</h3>", unsafe_allow_html=True)


def main():
    file_uploaded = st.file_uploader("Choose File", type=["jpg","jpeg"])
    class_btn = st.button("Classify")

    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)

    st.markdown("TILDI-Kaichon model by Kao Panboonyuen (~70.59% accuracy)")


def predict(image):
    classifier_model = "tildi-kaichon-v001.h5"
    IMAGE_SHAPE = (224, 224,3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
          'ไก่ด่างเบญจรงค์ (Benjarong Spotted)', 
          'ไก่ทองแดงหางดำ Tong Daeng Hang Dam (Copper with black tail)', 
          'ไก่นกกรดหางดำ Nok Kod Hang Dam (Reddish with Black tail)', 
          'ไก่นกแดง Nok Daeng (Red)', 
          'ไก่ประดู่หางดำ Pradoo Hang Dam (Partridge black red with black tail)', 
          'ไก่ประดู่เลาหางขาว Pradoo Hang Khao (Partridge black red with white tail)', 
          'ไก่เทาหางขาว Tao Hang Khao (Grey with white tail)', 
          'ไก่เหลืองหางขาว Luang Hang Khao (Yellow with white tail)']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'ไก่ด่างเบญจรงค์ (Benjarong Spotted)': 0, 
          'ไก่ทองแดงหางดำ Tong Daeng Hang Dam (Copper with black tail)': 1, 
          'ไก่นกกรดหางดำ Nok Kod Hang Dam (Reddish with Black tail)': 2, 
          'ไก่นกแดง Nok Daeng (Red)': 3, 
          'ไก่ประดู่หางดำ Pradoo Hang Dam (Partridge black red with black tail)': 4, 
          'ไก่ประดู่เลาหางขาว Pradoo Hang Khao (Partridge black red with white tail)': 5, 
          'ไก่เทาหางขาว Tao Hang Khao (Grey with white tail)': 6, 
          'ไก่เหลืองหางขาว Luang Hang Khao (Yellow with white tail)': 7
}

    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result

if __name__ == "__main__":
    main()


