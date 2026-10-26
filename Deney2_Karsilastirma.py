import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Sabitler ve Veri Hazırlığı
SEED = 42
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset, val_dataset = random_split(full_train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 2. Esnek MLP Sınıfı (Farklı katman sayılarını denemek için)
class MLP(nn.Module):
    def __init__(self, hidden1, hidden2):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(784, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 10)
        )
    def forward(self, x):
        return self.network(self.flatten(x))

# 3. Eğitim Döngüsü Fonksiyonu
def train_and_evaluate(model, optimizer, criterion, epochs=10):
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        train_loss = running_loss / len(train_dataset)
        
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_dataset)
        val_acc = (correct / len(val_dataset)) * 100
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} | Val Acc: %{val_acc:.2f} | Val Loss: {val_loss:.4f}")
        
    return history

print("--- 3 Farklı Senaryo Karşılaştırması Başlıyor ---")
EPOCHS = 10
criterion = nn.CrossEntropyLoss()

# Senaryo A: 256-128 Katman + Adam (Önceki en iyi modelimiz)
print("\n[Model A] 256-128 Katman + Adam Optimizer Eğitiliyor...")
model_a = MLP(256, 128).to(device)
opt_a = optim.Adam(model_a.parameters(), lr=0.001)
hist_a = train_and_evaluate(model_a, opt_a, criterion, EPOCHS)

# Senaryo B: 256-128 Katman + SGD (Adam ile SGD farkını görmek için)
print("\n[Model B] 256-128 Katman + SGD Optimizer Eğitiliyor...")
model_b = MLP(256, 128).to(device)
opt_b = optim.SGD(model_b.parameters(), lr=0.01, momentum=0.9)
hist_b = train_and_evaluate(model_b, opt_b, criterion, EPOCHS)

# Senaryo C: 64-32 Katman + Adam (Katman boyutu/nöron sayısı etkisini görmek için)
print("\n[Model C] 64-32 Katman (Küçük Ağ) + Adam Optimizer Eğitiliyor...")
model_c = MLP(64, 32).to(device)
opt_c = optim.Adam(model_c.parameters(), lr=0.001)
hist_c = train_and_evaluate(model_c, opt_c, criterion, EPOCHS)

# 4. Sonuçların Grafiğe Dökülmesi
epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(14, 6))

# Validation Accuracy Grafiği
plt.subplot(1, 2, 1)
plt.plot(epochs_range, hist_a['val_acc'], 'b-o', label='Model A (256-128, Adam)')
plt.plot(epochs_range, hist_b['val_acc'], 'r-s', label='Model B (256-128, SGD)')
plt.plot(epochs_range, hist_c['val_acc'], 'g-^', label='Model C (64-32, Adam)')
plt.title('Validation Doğruluk (Accuracy) Karşılaştırması')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Validation Loss Grafiği
plt.subplot(1, 2, 2)
plt.plot(epochs_range, hist_a['val_loss'], 'b-o', label='Model A (256-128, Adam)')
plt.plot(epochs_range, hist_b['val_loss'], 'r-s', label='Model B (256-128, SGD)')
plt.plot(epochs_range, hist_c['val_loss'], 'g-^', label='Model C (64-32, Adam)')
plt.title('Validation Kayıp (Loss) Karşılaştırması')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('deney2_karsilastirma.png', dpi=300)
plt.show()

print("\nKarşılaştırma tamamlandı. Grafik 'deney2_karsilastirma.png' olarak kaydedildi.")