import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load model
MODEL_PATH = 'cnn_classifier.h5'
model = load_model(MODEL_PATH)
class_names = ['Cats', 'Dogs']

# Page config
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐶", layout='centered')
st.title("# 🐾 Cat vs Dog Image Classifier")

st.markdown("""
A web app built using **Streamlit** and a **Vanilla CNN model** trained in TensorFlow/Keras.  
Upload an image of a cat or dog to get a prediction with confidence scores.
""")

# Sidebar: Upload & Sample
st.sidebar.header("📄 Upload or Choose Sample")
upload_file = st.sidebar.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

sample_type = st.sidebar.selectbox("Or select a sample", ["None", "Sample Cat", "Sample Dog"])
if sample_type != "None":
    sample_path = "sample_cat.jpg" if sample_type == "Sample Cat" else "sample_dog.jpg"
    img = Image.open(sample_path).convert('RGB')
    st.image(img, caption="🖼️ Sample Image", use_container_width=True)
    upload_file = sample_path  # Overwrite upload_file

# Show image immediately after upload
if upload_file and isinstance(upload_file, str) == False:
    img = Image.open(upload_file).convert('RGB')
    st.image(img, caption="🖼️ Your Uploaded Image", use_container_width=True)

# Prediction button state
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

if st.button("🔁 Run Prediction"):
    st.session_state.predict_clicked = True

if st.session_state.predict_clicked:
    if upload_file is None:
        st.warning("⚠️ Please upload or select an image before running prediction.")
        st.session_state.predict_clicked = False
    else:
        try:
            if isinstance(upload_file, str):
                img = Image.open(upload_file).convert('RGB')

            image_resized = img.resize((128, 128))
            img_array = image.img_to_array(image_resized) / 255.0
            image_batch = np.expand_dims(img_array, axis=0)

            prediction = model.predict(image_batch)
            predicted_class = class_names[np.argmax(prediction)]

            st.markdown(f"""
            <div style="padding:20px; border-radius:10px; background-color:#f0f2f6">
                <h3 style="color:#4CAF50">✅ Prediction: {predicted_class}</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Confidence Scores")
            col1, col2 = st.columns(2)
            for index, score in enumerate(prediction[0]):
                with (col1 if index % 2 == 0 else col2):
                    st.metric(label=class_names[index], value=f"{score * 100:.2f}%")
                    st.progress(int(score * 100))

            # Pie Chart
            st.markdown("### 📊 Visual Chart")
            fig, ax = plt.subplots()
            ax.pie(prediction[0], labels=class_names, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            st.session_state.predict_clicked = False

        except Exception as e:
            st.error("⚠️ Error processing the image.")
            st.exception(e)
            st.session_state.predict_clicked = False

# Feedback box
st.markdown("---")
feedback = st.text_input("💬 Any feedback about this prediction?")
if feedback:
    st.success("Thanks for your feedback!")

# About project section
with st.expander("ℹ️ About this project"):
    st.markdown("""
    - 📚 **Model:** Custom CNN trained on cats vs dogs dataset
    - 🧰 **Tech Stack:** TensorFlow, Keras, Streamlit, Python
    - 🧠 **Use Case:** Real-time image classification using computer vision
    - 📋 **GitHub:** [github.com/VikasKanwar101/Cnn_Image_Classifier](https://github.com/VikasKanwar101/Cnn_Image_Classifier)
    """)

# Add GitHub CTA below main content
st.markdown("---")
st.markdown("[📂 View Source Code on GitHub](https://github.com/VikasKanwar101/Cnn_Image_Classifier)")

# Contact info or credit
st.sidebar.markdown("---")
st.sidebar.markdown("**👨‍💻 Built by Vikas Kanwar**")
st.sidebar.markdown("[🐙 GitHub](https://github.com/VikasKanwar101) | [🔗 LinkedIn](https://www.linkedin.com/in/vikas-kanwar/)")
