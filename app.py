import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.title("MNIST Digit Predictor")
st.write("Upload an image of a handwritten digit to get a prediction.")

model_path = "67102010509_mnist_model.keras"


# Cache โมเดลไว้ในหน่วยความจำเพื่อไม่ต้องโหลดใหม่ทุกครั้งที่กดปุ่ม
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


if not os.path.exists(model_path):
    st.error(
        f"Model file '{model_path}' not found. Please ensure the model is saved correctly."
    )
else:
    model = load_model(model_path)

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Load และแสดงรูปภาพ
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)

            # ประมวลผลและทำนายผลด้วย st.spinner (จะซ่อนข้อความโหลดเมื่อทำงานเสร็จ)
            with st.spinner("Classifying..."):
                # Convert to grayscale
                img_gray = img.convert("L")

                # Resize to 28x28 pixels
                img_resized = img_gray.resize((28, 28))

                # Convert to numpy array
                img_array = np.array(img_resized)

                # Normalize pixel values [0, 255] -> [0, 1]
                img_array = img_array.astype("float32") / 255.0

                # Reshape (1, 28, 28) สำหรับส่งให้โมเดล
                img_array = img_array.reshape(1, 28, 28)

                # Make prediction
                prediction = model.predict(img_array)

                # ดึงตัวเลขที่มีความน่าจะเป็นสูงสุด และคำนวณ % ความมั่นใจ
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100

            # แสดงผลลัพธ์พร้อมความมั่นใจ (%)
            st.success(
                f"The model predicts the digit is: **{predicted_digit}** (Confidence: {confidence:.2f}%)"
            )

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
