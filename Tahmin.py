import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os

# 1. Model Mimarisi (Eğittiğimiz modelin iskeleti)
class BasicMLP(nn.Module):
    def __init__(self, input_size=784, hidden1=256, hidden2=128, num_classes=10):
        super(BasicMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )
    def forward(self, x):
        return self.network(self.flatten(x))

# 2. Kaydedilen Modeli Yükleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BasicMLP().to(device)

# Ağırlıkları diskten çağırıyoruz (Yeniden eğitim yok!)
model.load_state_dict(torch.load("egitilmis_mlp.pth", weights_only=True))
model.eval() # Modeli test moduna alıyoruz

# 3. Görüntü İşleme ve Tahmin Fonksiyonu
def resmi_hazirla_ve_tahmin_et(resim_yolu):
    if not os.path.exists(resim_yolu):
        print(f"{resim_yolu} bulunamadı!")
        return
        
    # Resmi yükle ve gri tonlamaya çevir
    img = Image.open(resim_yolu).convert('L')
    
    # Paint'te BEYAZ arka plana SİYAH kalemle çizdiğin için renkleri TERSLİYORUZ (Siyah arkaplan, beyaz yazı)
    img = ImageOps.invert(img)
    
    # 28x28 boyutuna küçült
    img = img.resize((28, 28))
    
    # Tensöre çevir ve normalize et
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Modele vermek için boyut ayarı yap: [1, 1, 28, 28]
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Modele tahmin ettir
    with torch.no_grad():
        cikti = model(img_tensor)
        _, tahmin = torch.max(cikti, 1)
        
    # Resmi ve Tahmini Ekranda Göster
    plt.imshow(img, cmap='gray')
    plt.title(f"Dosya: {resim_yolu} | Modelin Tahmini: {tahmin.item()}")
    plt.axis('off')
    plt.show()

# 4. Sol taraftaki 6 resmini sırayla test ediyoruz
resimler = ["rakam1.png", "rakam2.png", "rakam3.png", "rakam4.png", "rakam5.png", "rakam6.png","rakam7.png", "rakam8.png", "rakam9.png", "rakam10.png", "rakam11.png"]

for resim in resimler:
    resmi_hazirla_ve_tahmin_et(resim)