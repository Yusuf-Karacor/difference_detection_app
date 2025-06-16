📌 Proje Hakkında

    Bu masaüstü uygulaması, iki görüntü arasındaki farkları derin öğrenme ve geleneksel mesafe metriklerini kullanarak analiz etmek üzere geliştirilmiştir. Uygulama, Qt Designer ile oluşturulmuş bir arayüz üzerinden çalışmakta ve Python dili ile yazılmıştır.

🔧 Özellikler
  
    Kullanıcı, arayüz üzerinden iki adet görüntü yükleyebilir.
    
    Görseller arasında fark analizi için aşağıdaki yöntemlerden biri seçilebilir:
    
    Öklid (Euclidean) Mesafesi
    
    Kosinüs Benzerliği (Cosine Similarity)
    
    Manhattan Mesafesi
    
    SSIM (Structural Similarity Index)
    
    Derin özellik çıkarımı için iki farklı CNN modeli desteklenmektedir:
    
    ResNet152
    
    VGG19
    
    Kullanıcı tarafından eşik değeri belirlenebilir.
    
  Analiz sonucunda:
  
    İki görsel arasındaki fark, seçilen metrik ve model ile hesaplanır.
    
    Farklar heatmap (ısı haritası) olarak görselleştirilir.
    
    1 saniyelik aralıklarla iki görüntü arasında geçiş yapılarak dinamik bir karşılaştırma sunulur.

💻 Teknolojiler
    
    PyQt5 / Qt Designer: Arayüz tasarımı
    
    OpenCV: Görüntü işleme
    
    PyTorch: Derin öğrenme modelleri (VGG19, ResNet152)
    
    NumPy, Matplotlib: Sayısal işlemler ve görselleştirme

📂 Kurulum

    git clone https://github.com/Yusuf-Karacor/difference_detection_app
    
    cd difference_detection_app
    pip install -r requirements.txt
    python main_2.py


📷 Örnek Çıktılar

    Heatmap ile fark görselleştirmesi
    ![fark_çıktısı](https://github.com/user-attachments/assets/938d4671-10a1-4861-9b67-f7292bdf3bb2)

Model ve metrik seçimine göre değişen sonuçlar
