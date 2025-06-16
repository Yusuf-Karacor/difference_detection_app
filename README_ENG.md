<h1 align="center">🧠 Image Difference Detection Application</h1>

<p align="center">
  A desktop application that detects differences between two images using deep learning and classical distance metrics.
</p>

---

<h2 align="center">📌 About the Project</h2>

<p align="center">
This desktop application is designed to analyze differences between two images using deep learning and traditional similarity metrics. 
It features a GUI built with <strong>Qt Designer</strong> and is implemented in <strong>Python</strong>.
</p>

---

<h2 align="center">🔧 Features</h2>

<p align="center">
Users can load two images through the interface.
</p>

<p align="center">
The following comparison methods are available:
</p>

<p align="center">
  📏 Euclidean Distance<br/>
  📐 Cosine Similarity<br/>
  📊 Manhattan Distance<br/>
  🧩 SSIM (Structural Similarity Index)
</p>

<p align="center">
Supported CNN models for deep feature extraction:
</p>

<ul align="center">
  <li>🧠 ResNet152</li>
  <li>🧠 VGG19</li>
</ul>

<p align="center">
Users can also define a custom threshold value.
</p>

<ul align="center">
  <li>The difference between the images is calculated using the selected metric and model.</li>
  <li>The differences are visualized as a <strong>heatmap</strong>.</li>
  <li>Dynamic comparison is displayed by switching between images every 1 second.</li>
</ul>

---

<h2 align="center">💻 Technologies</h2>

<p align="center">
  <strong>PyQt5 / Qt Designer</strong> – GUI design<br/>
  <strong>OpenCV</strong> – Image processing<br/>
  <strong>PyTorch</strong> – Deep learning models (VGG19, ResNet152)<br/>
  <strong>NumPy, Matplotlib</strong> – Numerical operations & visualization
</p>

---

<h2 align="center">📂 Installation</h2>

```bash
git clone https://github.com/Yusuf-Karacor/difference_detection_app
cd difference_detection_app
pip install -r requirements.txt
python main_2.py
```

<h2 align="center">📷 Example Outputs</h2>

<p align="center">
<strong>Application Interface</strong><br/>
<img src="https://github.com/user-attachments/assets/319ed9da-768c-4fca-831a-bcf0342e00b5" width="600"/>
</p>

<p align="center">
<strong>Difference Visualization with Heatmap</strong><br/>
<img src="https://github.com/user-attachments/assets/938d4671-10a1-4861-9b67-f7292bdf3bb2" width="600"/>
</p>

<p align="center">
<strong>Results Vary Based on Selected Model and Metric</strong>
</p>

