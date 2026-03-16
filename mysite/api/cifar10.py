from fastapi import APIRouter, UploadFile, File, HTTPException
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

class CifarClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

cifar_router = APIRouter(prefix="/cifar", tags=["CIFAR10"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CifarClassification()
model.load_state_dict(torch.load("modelC.pth", map_location=device))
model.to(device)
model.eval()


@cifar_router.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Image can't be empty")

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = y_pred.argmax(dim=1).item()

        return {"Answer": CLASSES[pred]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))