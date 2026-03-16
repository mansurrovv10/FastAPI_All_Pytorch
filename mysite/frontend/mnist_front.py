import streamlit as st
import requests


def check_mnist():
    st.title("MNIST Classification with PyTorch")

    api_url = "http://127.0.0.1:8000/mnist/predict"

    uploaded_files = st.file_uploader(
        "Upload images", accept_multiple_files=True, type=["jpg", "png"]
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, width=200)

    if st.button("Predict"):

        if not uploaded_files:
            st.warning("Сүрөт жүктөңүз!")
        else:
            for uploaded_file in uploaded_files:

                response = requests.post(
                    api_url,
                    files={
                        "image": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: **{result['Answer']}**")
                else:
                    st.error("API error")