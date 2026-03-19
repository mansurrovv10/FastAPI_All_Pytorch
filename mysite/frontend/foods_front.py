import streamlit as st
import requests

def check_foods():
    st.title("Foods With Pytorch")
    st.text("Загрузите Изоброжение ")

    st.text('Загрузить изображение с цифровой,и модел попробует ее распознать ')
    api_url = "http://127.0.0.1:8000/foods/predict"

    uploaded_files = st.file_uploader(
        "Upload Foods Image",accept_multiple_files=True, type=["jpg", "png"]
    )


    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.image(uploaded_file)


    if st.button("Predict"):

        for uploaded_file in uploaded_files:


            files = {"image": uploaded_file.getvalue()}

            response = requests.post(api_url, files={"image": uploaded_file})


            if response.status_code == 200:
                result = response.json()
                st.success(f"Модел думает что это : {result['Answer']}")
            else :
                st.error("API Error")