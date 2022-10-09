import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io

model= keras.models.load_model('dog_classif_model.h5',compile=False)
class_names=np.genfromtxt('class_names.csv',delimiter=',',dtype='str')
def main():
    st.title('Know your dog\'s breed')
    st.header('Description:')
    st.write('This streamlit app is used for predicting a dog\'s breed with deep learning model. The model was made using transfer learning from Xception pretrained model and retrained/tuned on StanfordDogs dataset. It reached 84% TOP 1 accuracy and 97% TOP 5 accuracy on the test set.')
    st.write('You can find the whole project on my [repo](https://github.com/zakaria-narjis/Know-your-dog-s-breed).')
    st.header('Dog breed classification:')
    load_image()

def load_image():
    uploaded_file = st.file_uploader(label='Upload the image of your dog')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        top1,top5=predict(image_data)
        st.subheader('Your dog\'s breed TOP 1 prediction:')
        st.write(f'{top1[0]} , Probability of {top1[1]*100} %')
        st.subheader('Your dog\'s breed TOP 5 prediction:')
        #for label in top5:
            #st.write(f'{label[0]} , Probability of {label[1] * 100} %')
        data_={'Label':[class_[0] for class_ in top5], 'Probability %': [class_[1]*100 for class_ in top5]}
        top5_data=pd.DataFrame(data=data_)
        st.table(top5_data)
        
def predict(image):
    image=Image.open(io.BytesIO(image)).convert('RGB')
    img_to_tensor = tf.keras.preprocessing.image.img_to_array(image)
    img=preprocess(img_to_tensor)
    prediction = model(img.numpy().reshape(1, 299, 299, 3), training=False)
    top1_class=(class_names[np.argmax(prediction.numpy()[0])],np.amax(prediction))
    top5_class =[]
    for index in np.argpartition(prediction.numpy()[0], -5)[-5:]:
        top5_class.append((class_names[index],prediction.numpy()[0][index]))
    return top1_class,top5_class

def preprocess(image):
    resized_image = tf.image.resize(image, [299, 299])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image

if __name__=='__main__':
    main()
