from fastapi import APIRouter, UploadFile, File, HTTPException
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x



transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

mnist_router = APIRouter(prefix="/mnist", tags=["MNIST"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CheckImage()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()


@mnist_router.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Image can't be empty")

        img = Image.open(io.BytesIO(image_bytes))

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = y_pred.argmax(dim=1).item()

        return {"Answer": pred}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

