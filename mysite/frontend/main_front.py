import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from mnist_front import check_mnist
from fashion_front import check_fashion
from cifar10_front import check_cifar10
from flowers_front import check_flowers
from foods_front import check_foods
with st.sidebar:
    st.title("DL MODELS")
    name = st.radio("Choose a dataset", ["Info","MNIST","FashionMNIST", "CIFAR10", "Flowers","Foods"])


if name == "Info":
   st.title("Welcome")
   st.text("MNIST - распознавание цифры")
   st.text("Fashion MNIST - классификация одежды")
   st.text("CIFAR10 - классификация изображений")
   st.text("Flowers - классификация изображений с цветы")
   st.text("Foods - классификация изображений с едой")

elif name == "MNIST":
    check_mnist()

elif name == "FashionMNIST":
    check_fashion()

elif name == "CIFAR10":
    check_cifar10()

elif name == "Flowers":
    check_flowers()

elif name == "Foods":
    check_foods()



