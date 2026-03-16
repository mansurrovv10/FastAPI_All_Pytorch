import streamlit as st
import requests

def check_cifar10():
    st.title("CIFAR10 Image Classification")

    api_url = "http://127.0.0.1:8000/cifar/predict"

    uploaded_files = st.file_uploader(
        "Upload images", accept_multiple_files=True, type=["jpg", "jpeg", "png"]
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.image(uploaded_file)

    if st.button("Определить изображение"):
        if not uploaded_files:
            st.warning("Сүрөт жүктөңүз!")
        else:
            for uploaded_file in uploaded_files:
                try:
                    response = requests.post(
                        api_url,
                        files={"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Модель думает что это: **{result['Answer']}**")
                    else:
                        st.error(f"Ошибка API: {response.status_code} — {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("FastAPI сервер иштебей жатат!")

                except Exception as e:
                    st.error(f"Ката: {e}")