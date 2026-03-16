from fastapi import APIRouter, UploadFile, File, HTTPException
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class CheckDress(nn.Module):
    def __init__(self):
        super().__init__()

        self.one= nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.two = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*14*14,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
    def forward(self, x):
            x = self.one(x)
            x = self.two(x)
            return x


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

fashion_router = APIRouter(prefix="/fashion", tags=["Fashion MNIST"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CheckDress()
model.load_state_dict(torch.load("model (1).pth", map_location=device))
model.to(device)
model.eval()


forms = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',

    8: 'Bag',
    9: 'Ankle boot'
}

@fashion_router.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Сүрөт бош болбошу керек")

        img = Image.open(io.BytesIO(image_bytes))

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred_index = y_pred.argmax(dim=1).item()
            pred_label = forms.get(pred_index, "Белгисиз")

        return {"Answer": pred_label}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
