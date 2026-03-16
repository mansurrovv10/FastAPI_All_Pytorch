from fastapi import FastAPI
import uvicorn
from mysite.api import mnist,cifar10,fashion


pytorch_app = FastAPI()
pytorch_app.include_router(mnist.mnist_router)
pytorch_app.include_router(fashion.fashion_router)
pytorch_app.include_router(cifar10.cifar_router)


if __name__ == "__main__":
    uvicorn.run(pytorch_app, host="127.0.0.1", port=8000)