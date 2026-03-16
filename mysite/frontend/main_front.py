import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from mnist_front import check_mnist
from fashion_front import check_fashion
from cifar10_front import check_cifar10


with st.sidebar:
    st.title("MNIST,Fashion MNIST,CIFAR10")
    name = st.radio("Choose a dataset", ["Info","MNIST","FashionMNIST", "CIFAR10"])


if name == "Info":
   st.title("Welcome")
   st.text("MNIST - распознавание цифры")
   st.text("Fashion MNIST - классификация одежды")
   st.text("CIFAR10 - классификация изображений")

elif name == "MNIST":
    check_mnist()

elif name == "FashionMNIST":
    check_fashion()

elif name == "CIFAR10":
    check_cifar10()



