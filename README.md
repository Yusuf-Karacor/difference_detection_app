<h1 align="center">🧠 Görüntü Farkı Tespit Uygulaması</h1>

<p align="center">
  Derin öğrenme ve görüntü işleme teknikleriyle, iki görsel arasındaki farkları tespit eden masaüstü uygulaması.
</p>

---

<h2 align="center">📌 Proje Hakkında</h2>

<p align="center">
Bu masaüstü uygulaması, iki görüntü arasındaki farkları derin öğrenme ve geleneksel mesafe metriklerini kullanarak analiz etmek üzere geliştirilmiştir. 
Uygulama, <strong>Qt Designer</strong> ile oluşturulmuş bir arayüz üzerinden çalışmakta ve <strong>Python</strong> dili ile yazılmıştır.
</p>

---

<h2 align="center">🔧 Özellikler</h2>

<p align="center">
Kullanıcı, arayüz üzerinden iki adet görüntü yükleyebilir.
</p>

<p align="center">
Görseller arasında fark analizi için aşağıdaki yöntemlerden biri seçilebilir:
</p>

<p align="center">
  📏 Öklid (Euclidean) Mesafesi</li>
  📐 Kosinüs Benzerliği (Cosine Similarity)</li>
  📊 Manhattan Mesafesi</li>
  🧩 SSIM (Structural Similarity Index)</li>
</p>

<p align="center">
Derin özellik çıkarımı için desteklenen CNN modelleri:
</p>

<ul align="center">
  <li>🧠 ResNet152</li>
  <li>🧠 VGG19</li>
</ul>

<p align="center">
Kullanıcı tarafından eşik değeri belirlenebilir. Analiz sonucunda:
</p>

<ul align="center">
  <li>İki görsel arasındaki fark, seçilen metrik ve model ile hesaplanır.</li>
  <li>Farklar <strong>heatmap (ısı haritası)</strong> olarak görselleştirilir.</li>
  <li>1 saniyelik aralıklarla görseller arasında geçiş yapılır.</li>
</ul>

---

<h2 align="center">💻 Teknolojiler</h2>

<p align="center">
  <strong>PyQt5 / Qt Designer</strong>: Arayüz tasarımı<br/>
  <strong>OpenCV</strong>: Görüntü işleme<br/>
  <strong>PyTorch</strong>: Derin öğrenme modelleri (VGG19, ResNet152)<br/>
  <strong>NumPy, Matplotlib</strong>: Sayısal işlemler ve görselleştirme
</p>

---

<h2 align="center">📂 Kurulum</h2>

```bash
git clone https://github.com/Yusuf-Karacor/difference_detection_app
cd difference_detection_app
pip install -r requirements.txt
python main_2.py
```

<p align="center">

📷 Örnek Çıktılar
![ara_yüz](https://github.com/user-attachments/assets/319ed9da-768c-4fca-831a-bcf0342e00b5)

Heatmap ile fark görselleştirmesi
![fark_çıktısı](https://github.com/user-attachments/assets/938d4671-10a1-4861-9b67-f7292bdf3bb2)

Model ve metrik seçimine göre değişen sonuçlar
</p>

