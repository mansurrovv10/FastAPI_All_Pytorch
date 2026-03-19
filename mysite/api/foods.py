import io
from sys import prefix

from fastapi import APIRouter,UploadFile,File,HTTPException
import uvicorn
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image


class CheckFood(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

foods_router = APIRouter(prefix='/foods',tags=['Foods'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = CheckFood().to(device)
model.load_state_dict(torch.load("modelFoods.pth", map_location=device))
model.to(device)
model.eval()

class_names = ['burger', 'cake', 'pizza', 'salad', 'sushi']

@foods_router.post('/predict')
async def predict(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Сүрөт бош болбошу керек")

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred_index = y_pred.argmax(dim=1).item()
            pred_label = class_names[pred_index]

        return {"Answer": pred_label}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
