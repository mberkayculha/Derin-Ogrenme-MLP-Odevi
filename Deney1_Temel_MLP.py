import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Tekrarlanabilirlik (Reproducibility) için rastgelelik tohumunu sabitliyoruz
SEED = 42
torch.manual_seed(SEED)

# 2. Bilgisayarda veya Colab'de ekran kartı (GPU) varsa onu, yoksa işlemciyi (CPU) seçiyoruz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")

# 1. Veri Ön İşleme (Preprocessing) ve Normalizasyon Kuralları
transform = transforms.Compose([
    transforms.ToTensor(), # Görüntüyü [0, 255] aralığından [0.0, 1.0] aralığında PyTorch tensörüne çevirir
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST'in ortalama (mean) ve standart sapması (std) ile normalize eder
])

# 2. Orijinal MNIST Veri Setini İndirme
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 3. Train (50.000) ve Validation (10.000) Ayrımı
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(
    full_train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED) # Ayrımın her seferinde aynı olması için
)

# 4. Veri Yükleyicilerin (DataLoader) Oluşturulması
BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train veri sayısı      : {len(train_dataset)}")
print(f"Validation veri sayısı : {len(val_dataset)}")
print(f"Test veri sayısı       : {len(test_dataset)}")

class BasicMLP(nn.Module):
    def __init__(self, input_size=784, hidden1=256, hidden2=128, num_classes=10):
        super(BasicMLP, self).__init__()
        
        # 28x28 boyutundaki 2B görüntüyü 784 elemanlı 1B düz vektöre çevirir
        self.flatten = nn.Flatten()
        
        # Katmanları sırasıyla diziyoruz
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1), # 1. Gizli Katman: 784 girdi -> 256 nöron
            nn.ReLU(),                      # Aktivasyon: Negatifleri sıfırla
            nn.Linear(hidden1, hidden2),    # 2. Gizli Katman: 256 girdi -> 128 nöron
            nn.ReLU(),                      # Aktivasyon: Negatifleri sıfırla
            nn.Linear(hidden2, num_classes) # Çıktı Katmanı: 128 girdi -> 10 sınıf (0-9 rakamları)
        )
        
    def forward(self, x):
        x = self.flatten(x)      # Önce resmi düzleştir: [Batch, 1, 28, 28] -> [Batch, 784]
        logits = self.network(x) # Katmanlardan geçir ve 10 sınıf için ham skorları (logits) üret
        return logits

# Modeli oluşturup GPU/CPU'ya gönderiyoruz
model = BasicMLP().to(device)
print(model)

# Hiperparametreler (Raporda belirtilmesi zorunlu)
LEARNING_RATE = 0.001
EPOCHS = 15

# Kayıp Fonksiyonu (Loss) ve Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # Modeli eğitim moduna alır
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # 1. Önceki adımdan kalan gradyanları (türevleri) sıfırla
        optimizer.zero_grad()
        
        # 2. İleri Yayılım (Forward Pass): Model tahmin üretsin
        outputs = model(images)
        
        # 3. Kaybı (Loss) hesapla
        loss = criterion(outputs, labels)
        
        # 4. Geriye Yayılım (Backward Pass): Hatanın ağırlıklara göre türevini al
        loss.backward()
        
        # 5. Ağırlıkları güncelle
        optimizer.step()
        
        # İstatistikleri topla
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1) # En yüksek skoru alan sınıfı seç
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval() # Modeli değerlendirme moduna alır
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Değerlendirme sırasında ağırlık güncellemeyeceğimiz için türev hesabını kapatıyoruz (Hız ve bellek tasarrufu)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

# Grafik çizimi için her epoch sonucunu saklayacağımız listeler
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': []
}

print("Eğitim Başlıyor...\n" + "-"*65)
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | "
          f"Train Loss: {train_loss:.4f}, Train Acc: %{train_acc:.2f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: %{val_acc:.2f}")

total_time = time.time() - start_time
print("-" * 65)
print(f"Toplam Eğitim Süresi: {total_time:.2f} saniye")

# EĞİTİM BİTTİKTEN SONRA NİHAİ TEST DEĞERLENDİRMESİ (Kural 5: Sadece en sonda 1 kez!)
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Nihai Test Loss: {test_loss:.4f} | Nihai Test Accuracy: %{test_acc:.2f}")

# --- GRAFİKLERİN ÇİZDİRİLMESİ ---
epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(14, 5))

# 1. Loss Grafiği
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], 'b-o', label='Training Loss')
plt.plot(epochs_range, history['val_loss'], 'r-s', label='Validation Loss')
plt.title('Deney 1: Eğitim ve Doğrulama Kaybı (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 2. Accuracy Grafiği
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_acc'], 'b-o', label='Training Accuracy')
plt.plot(epochs_range, history['val_acc'], 'r-s', label='Validation Accuracy')
plt.title('Deney 1: Eğitim ve Doğrulama Doğruluğu (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('deney1_grafikler.png', dpi=300) # GitHub raporu için yüksek kalitede kaydeder
plt.show()

# Modeli kaydet (Bu sayede bir daha asla eğitmeyeceğiz)
torch.save(model.state_dict(), "egitilmis_mlp.pth")