import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os

# CUDA (GPU) Kontrolü: Eğer NVIDIA ekran kartı varsa işlemler 10-50 kat hızlanır.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =============================================================================
# 1. BÖLÜM: VERİ ÖN İŞLEME (PANDAS & SKLEARN)
# =============================================================================
def load_and_preprocess_data(file_path='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Bu fonksiyon ham veriyi alır ve yapay sinir ağının anlayacağı 
    matematiksel formata (matrislere) çevirir.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'{file_path}' dosyası bulunamadı!")
    
    df = pd.read_csv(file_path)
    
    # --- Adım 1: Gereksiz Sütun Temizliği ---
    # customerID: Müşterinin kimlik nosunun churn (terk etme) ile bir alakası yoktur.
    # Bu tarz 'unique' (eşsiz) tanımlayıcılar modelin ezber yapmasına sebep olabilir, atılmalıdır.
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    # --- Adım 2: Veri Tipi Düzeltme ---
    # 'TotalCharges' sütunu bazen string (metin) olarak gelir çünkü içinde boşluklar (" ") olabilir.
    # errors='coerce': Hatalı (sayı olmayan) değerleri NaN (boş veri) yap demektir.
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # NaN olan satırları atıyoruz (Genelde çok azdır, veri kaybı önemsizdir)
    df.dropna(inplace=True)
    
    # --- Adım 3: Kategorik Verileri Sayısallaştırma (EN ÖNEMLİ KISIM) ---
    
    # YÖNTEM A: Label Encoding (0, 1)
    # Sadece 2 seçeneği olan (Evet/Hayır, Kadın/Erkek) veriler için uygundur.
    le = LabelEncoder()
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                   'PaperlessBilling', 'Churn']
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    
    # YÖNTEM B: One-Hot Encoding (Dummy Variables)
    # İkiden fazla seçeneği olanlar için (Örn: İnternet Tipi -> DSL, Fiber, Yok).
    # Neden Label Encoder kullanmadık?
    # Cevap: Eğer DSL=0, Fiber=1, Yok=2 dersek; model "Yok, DSL'den büyüktür" gibi 
    # saçma bir matematiksel ilişki kurar. One-Hot bunu engeller, her seçeneği ayrı sütun yapar.
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                        'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaymentMethod']
    
    # get_dummies: Kategorik sütunları 0 ve 1'lerden oluşan yeni sütunlara böler.
    # drop_first=False: Tüm sınıfları tutuyoruz. (Bazen 'Linear Regression'da tuzağa düşmemek için True yapılır ama YSA'da şart değildir)
    if categorical_cols:
        existing_cat_cols = [col for col in categorical_cols if col in df.columns]
        df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=False)
    
    # --- Adım 4: Özellikler (X) ve Hedef (y) Ayrımı ---
    # .values: Pandas DataFrame'i Numpy dizisine çevirir. PyTorch Numpy sever.
    X = df.drop('Churn', axis=1).values
    y = df['Churn'].values
    
    return X, y

# =============================================================================
# 2. BÖLÜM: VERİ HAZIRLAMA (PYTORCH DATA PIPELINE)
# =============================================================================
def prepare_data(X, y, test_size=0.2, batch_size=32):
    
    # Soru 1: stratify=y parametresi neden önemli?
    # Cevap 1: Churn verisetlerinde genelde 'Terk Edenler' az, 'Kalanlar' çoktur (Dengesiz Veri).
    # Stratify, eğitim ve test setinde bu oranın (örn: %20 Churn, %80 Kalan) korunmasını sağlar.
    # Aksi takdirde test setine hiç Churn eden müşteri düşmeyebilir ve test yanıltıcı olur.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # --- Adım 2: Ölçeklendirme (Scaling) ---
    # Soru 2: Neden StandardScaler (Normalizasyon) şart?
    # Cevap 2: Bir sütunda "Aylık Ücret" (örn: 70.5), diğerinde "Hizmet Süresi" (örn: 72 ay) var.
    # Neural Network, büyük sayıları daha "önemli" zanneder ve ağırlıkları (weights) dengesiz günceller.
    # Ölçekleme, tüm verileri yaklaşık -1 ile +1 arasına çeker, eğitimin hızlı ve kararlı olmasını sağlar.
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train) # Eğitim setinden öğren ve dönüştür
    X_test = sc.transform(X_test)       # Sadece dönüştür (Test setinden öğrenmek hiledir!)
    
    # --- Adım 3: Tensor Dönüşümü ---
    # PyTorch sadece 'Tensor' adı verilen özel matrislerle çalışır.
    # FloatTensor: Ondalıklı sayılar için standarttır.
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    
    # Soru 3: y_train.reshape(-1, 1) ne işe yarıyor?
    # Cevap 3: y_train normalde [0, 1, 0, ...] şeklindedir (Boyut: N).
    # Ancak Model çıkışı [[0.2], [0.9], ...] şeklindedir (Boyut: N x 1).
    # Loss fonksiyonunun hata vermemesi için boyutların birebir eşleşmesi gerekir.
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # --- Adım 4: DataLoader ---
    # Veriyi mini-batch'lere (küçük paketlere) böler.
    # shuffle=True: Modelin verinin sırasını ezberlemesini önlemek için her epoch'ta karıştırır.
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# =============================================================================
# 3. BÖLÜM: MODEL MİMARİSİ (YAPAY SİNİR AĞI)
# =============================================================================
class ChurnANN(nn.Module):
    def __init__(self, input_size):
        super(ChurnANN, self).__init__()
        
        # Katman 1 (Giriş -> Gizli): 
        # input_size: Verideki sütun sayısı (Örn: 40 özellik)
        # 16: Bizim seçtiğimiz nöron sayısı (Deneme yanılma ile bulunur)
        self.fc1 = nn.Linear(input_size, 16)
        
        # Katman 2 (Gizli -> Gizli):
        # Derinlik kazandırmak için eklenir.
        self.fc2 = nn.Linear(16, 8)
        
        # Katman 3 (Gizli -> Çıkış):
        # Çıktı 1 nöron olmak zorunda çünkü sonuç tek bir ihtimal: Churn mü (1) değil mi (0)?
        self.fc3 = nn.Linear(8, 1)
        
        # Aktivasyonlar
        self.relu = nn.ReLU()     # Negatifleri sıfırlar, lineerliği bozar (Öğrenmeyi sağlar)
        self.sigmoid = nn.Sigmoid() # Sonucu 0 ile 1 arasına sıkıştırır (Olasılık üretir)
    
    def forward(self, x):
        # Verinin akış yönü:
        x = self.relu(self.fc1(x))  # Girdi -> Linear -> ReLU
        x = self.relu(self.fc2(x))  # -> Linear -> ReLU
        x = self.sigmoid(self.fc3(x)) # -> Linear -> Sigmoid -> Olasılık (0.85 gibi)
        return x

# =============================================================================
# 4. BÖLÜM: OPTİMİZASYON AYARLARI
# =============================================================================
def define_loss_and_optimizer(model, lr=0.01):
    
    # Soru 4: Neden BCELoss ve CrossEntropyLoss değil?
    # Cevap 4: 
    # - BCELoss (Binary Cross Entropy): İkili sınıflandırma (0 veya 1) içindir ve modelin ucunda Sigmoid varsa kullanılır.
    # - CrossEntropyLoss: Çoklu sınıflandırma (Kedi, Köpek, Kuş) içindir ve içinde kendi Softmax'ı vardır.
    # Bizim modelin sonunda Sigmoid olduğu için BCELoss şarttır.
    criterion = nn.BCELoss()
    
    # Adam: En popüler optimizasyon algoritmasıdır. Learning Rate'i (lr) duruma göre kendi ayarlar.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return criterion, optimizer

# =============================================================================
# 5. BÖLÜM: EĞİTİM DÖNGÜSÜ (THE TRAINING LOOP)
# =============================================================================
def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train() # Modeli eğitim moduna al (Dropout vs. varsa aktifleşir)
    loss_history = []
    
    print("\n--- EĞİTİM BAŞLIYOR ---")
    for epoch in range(epochs):
        running_loss = 0.0
        
        for features, labels in train_loader:
            # Veriyi GPU'ya gönder (Eğer varsa)
            features, labels = features.to(device), labels.to(device)
            
            # 1. Gradyanları sıfırla (Eski hesaplamaları unut)
            optimizer.zero_grad()
            
            # 2. İleri Yayılım (Tahmin yap)
            outputs = model(features)
            
            # 3. Hatayı Hesapla
            loss = criterion(outputs, labels)
            
            # 4. Geri Yayılım (Hatanın kaynağını bul - Türev al)
            loss.backward()
            
            # 5. Ağırlıkları Güncelle
            optimizer.step()
            
            running_loss += loss.item()
        
        # Her epoch sonu ortalama hatayı kaydet
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    return loss_history

# =============================================================================
# 6. BÖLÜM: TEST VE DEĞERLENDİRME
# =============================================================================
def test_model(model, test_loader):
    model.eval() # Modeli test moduna al (Dropout kapanır, BatchNorm sabitlenir)
    all_preds = []
    all_labels = []
    
    print("\n--- TEST SONUÇLARI ---")
    
    # with torch.no_grad(): Gradyan hesaplama, hafıza harcama! Sadece tahmin yapıyoruz.
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features) # Çıktılar 0.2, 0.8 gibi olasılıklardır.
            
            # 0.5'ten büyükse 1 (Churn), küçükse 0 (No Churn) yap
            predictions = outputs.round() 
            
            # Veriyi CPU'ya alıp Numpy formatına çevirmeliyiz (Sklearn GPU tensorü kabul etmez)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Doğruluk (Accuracy): Toplam doğru tahmin / Toplam veri
    accuracy = (all_preds == all_labels).mean() * 100
    print(f'Test Doğruluğu: %{accuracy:.2f}')
    
    # Confusion Matrix: Nerede hata yaptık?
    # [[Gerçek Negatif, Yalancı Pozitif],
    #  [Yalancı Negatif, Gerçek Pozitif]]
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Detaylı Rapor (Precision, Recall, F1-Score)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=['No Churn', 'Churn']))
    
    return accuracy

# =============================================================================
# 7. BÖLÜM: GÖRSELLEŞTİRME
# =============================================================================
def plot_loss(loss_vals, epochs):
    """
    Eğitim kaybının düşüşünü gösteren grafik.
    Sağ aşağı doğru inen bir kaydırak görmeliyiz.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_vals, label="Training Loss", linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# =============================================================================
# ANA ÇALIŞTIRMA BLOĞU
# =============================================================================
if __name__ == "__main__":
    # 1. Veriyi Yükle
    # NOT: Bu dosya dizinde olmalı.
    X, y = load_and_preprocess_data(file_path='WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # 2. Hazırla
    train_loader, test_loader = prepare_data(X, y, test_size=0.2, batch_size=32)
    
    # 3. Modeli Kur
    input_size = X.shape[1] # One-Hot sonrası özellik sayısı (Örn: 40'a çıkmış olabilir)
    model = ChurnANN(input_size).to(device)
    print(f"\nModel Mimarisi:\n{model}")
    
    # 4. Ayarları Yap
    criterion, optimizer = define_loss_and_optimizer(model, lr=0.01)
    
    # 5. Eğit
    total_epochs = 100
    loss_vals = train_model(model, train_loader, criterion, optimizer, epochs=total_epochs)
    
    # 6. Grafiği Çiz
    plot_loss(loss_vals, total_epochs)
    
    # 7. Test Et
    test_model(model, test_loader)