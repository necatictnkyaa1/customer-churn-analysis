# 📉 Customer Churn Analysis with PyTorch

Bu proje, telekomünikasyon verilerini kullanarak müşterilerin hizmeti terk edip etmeyeceğini (Churn) tahmin eden bir **Yapay Sinir Ağı (Artificial Neural Network - ANN)** modelidir.

Proje, veri ön işleme (preprocessing), PyTorch ile model eğitimi ve performans değerlendirmesi adımlarını içerir.

## 🚀 Özellikler

- **Veri Analizi:** Pandas ile veri manipülasyonu ve temizliği.
- **Ön İşleme:** Eksik verilerin yönetimi, Label Encoding ve One-Hot Encoding işlemleri.
- **Dengesiz Veri Yönetimi:** `stratify` parametresi ile dengeli train/test ayrımı.
- **Derin Öğrenme:** PyTorch kullanılarak oluşturulmuş çok katmanlı (Multi-Layer Perceptron) mimari.
- **Modern Paket Yönetimi:** Proje bağımlılıkları `uv` ile yönetilmektedir.

## 🛠️ Kullanılan Teknolojiler

* **Dil:** Python 3.10+
* **Derin Öğrenme:** PyTorch
* **Veri İşleme:** Pandas, NumPy
* **Makine Öğrenmesi (Ön işleme):** Scikit-Learn
* **Görselleştirme:** Matplotlib

## 📂 Proje Yapısı

* `load_and_preprocess_data`: Veriyi yükler, temizler ve matris formatına çevirir.
* `prepare_data`: Veriyi eğitim ve test setlerine ayırır, ölçeklendirir (StandardScaler) ve Tensor'a dönüştürür.
* `ChurnANN`: PyTorch tabanlı yapay sinir ağı sınıfı.
* `train_model`: Eğitim döngüsü (Training Loop).
* `test_model`: Modelin başarısını ölçer (Confusion Matrix & Accuracy).

## 💻 Kurulum ve Çalıştırma

Bu proje modern Python paket yöneticisi **uv** kullanılarak hazırlanmıştır.

1. **Repoyu klonlayın:**
   ```bash
   git clone [https://github.com/necatictnkyaa1/customer-churn-analysis.git](https://github.com/necatictnkyaa1/customer-churn-analysis.git)
   cd customer-churn-analysis
