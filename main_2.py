import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QLabel
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage,QDesktopServices
import cv2
from PIL import Image
import numpy as np  # NumPy (sayısal işlemler için)
import torch  # PyTorch (derin öğrenme kütüphanesi)
import torchvision.transforms as transforms  # PyTorch görüntü dönüşümleri
from torchvision import models  # PyTorch önceden eğitilmiş modeller
import seaborn as sns  # Heatmap için Seaborn kütüphanesi
from datetime import datetime
import os
from PyQt5.QtCore import QStandardPaths,QObject,QTimer
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt,QUrl
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np



class FarkWorker(QObject):
    # finished = pyqtSignal()
    # progress = pyqtSignal(str)
    # stopped = False  # İptal için kontrol

    def get_patches_for_both_images(self,img1, img2, grid_size):
        patches1 = []  # İlk resmin yamaları
        patches2 = []  # İkinci resmin yamaları
        
        # Görüntü boyutlarını al
        h, w, _ = img1.shape  # Görüntü boyutları
        patch_h, patch_w = h // grid_size, w // grid_size  # Yama boyutları

        # Görüntüleri grid_size x grid_size boyutunda parçalara ayır
        for i in range(grid_size):
            for j in range(grid_size):
                patch1 = img1[i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]  # img1'den yama çıkar
                patch2 = img2[i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]  # img2'den yama çıkar
                
                patches1.append(patch1)
                patches2.append(patch2)

        return patch_h, patch_w,patches1, patches2

    def align_and_crop_images(self,img1, img2):
        # Görüntü hizalama işlemi (ORB + Homografi kullanarak)
        orb = cv2.ORB_create()  # ORB nesnesi oluştur
        kp1, des1 = orb.detectAndCompute(img1, None)  # Anahtar noktaları ve tanımlayıcıları bul
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            QMessageBox.warning(self, "Hata", "Anahtar nokta tanımlayıcıları bulunamadı. Lütfen farklı görüntüler deneyin.")


        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Brute-Force Matcher nesnesi oluştur
        matches = bf.match(des1, des2)  # Eşleşmeleri bul
        matches = sorted(matches, key=lambda x: x.distance)  # Eşleşmeleri sırala
        if len(matches) < 4:
            QMessageBox.warning(self, "Hata", "YETERSİZ EŞLEŞME RESİMLERİ KONTROL EDİN")

        # Kaynak ve hedef noktalarını oluştur
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Homografi matrisini hesapla ve görüntüyü hizala
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        aligned_img2 = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

        # Siyah alanları temizle (gri tonlamalıya çevir ve eşikleme yap)
        gray = cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # Siyah olmayan bölgeleri beyaz yap

        # Beyaz alanların etrafındaki dikdörtgeni bul
        x, y, w, h = cv2.boundingRect(thresholded)

        # Görüntüyü dikdörtgen bölgeye göre kırp
        cropped_img2 = aligned_img2[y+25:y+h-25, x+25:x+w-25]  # img2'yi kırp
        cropped_img1 = img1[y+25:y+h-25, x+25:x+w-25] # img1'i kırp
        cropped_img1=cv2.resize(cropped_img1,(1080,720))
        cropped_img2=cv2.resize(cropped_img2,(1080,720))

        return cropped_img1, cropped_img2

    def oklid_and_manhattan_def(self,patches1, patches2, adet, transform, patch_w, patch_h ,esik_manhattan,esik_oklid):  
        print("OKLİD_AND_MANHATTAN içinde")

        diff_scores_ok = []  # Fark skorları listesi
        diff_scores_mn = []  # Fark skorları listesi

        for p1, p2 in zip(patches1, patches2):  # Yamaları eşleştir
            p1 = transform(Image.fromarray(p1)).unsqueeze(0).to('cuda')   # Tensöre dönüştür ve boyut ekle
            p2 = transform(Image.fromarray(p2)).unsqueeze(0).to('cuda') 
            # Yamaları tensörlere dönüştürüyoruz ve modelin beklediği boyuta getiriyoruz.

            with torch.no_grad():  # Gradyanları hesaplama
                feat1 = model(p1).flatten().cpu().numpy()  # Özellik vektörünü çıkar, önce CPU'ya taşı, sonra NumPy'ye çevir
                feat2 = model(p2).flatten().cpu().numpy()  # Aynı işlemi diğer yama için de yap

            euclidean_distance = np.linalg.norm(feat1 - feat2)  # Öklid uzaklığı hesapla
            diff_scores_ok.append(euclidean_distance)  # Uzaklık değerini listeye ekle
            manhattan_distance = np.sum(np.abs(feat1 - feat2)) # Manhattan mesafesini kullan
            diff_scores_mn.append(manhattan_distance)  # Uzaklık değerini listeye ekle

        diff_matrix_mn = np.array(diff_scores_mn).reshape(adet, adet)

        diff_matrix_ok = np.array(diff_scores_ok).reshape(adet, adet)

        kareler_eu = []
        kareler_mn = []
        sonuc = []

        for i in range(adet):
            for j in range(adet):
                if diff_matrix_ok[i, j] > esik_oklid:  # Eşik değeri aşan farklar
                    x1, y1 = j * patch_w, i * patch_h
                    x2, y2 = (j + 1) * patch_w, (i + 1) * patch_h
                    kareler_eu.append((x1,y1,x2,y2))
                if diff_matrix_mn[i, j] > esik_manhattan:  # Eşik değerinden büyükse
                    x1, y1 = j * patch_w, i * patch_h  # Kare koordinatları
                    x2, y2 = (j + 1) * patch_w, (i + 1) * patch_h
                    kareler_mn.append((x1, y1, x2, y2)) 

        for kare_cos in kareler_eu:
            for kare_ok in kareler_mn:
                if kare_cos == kare_ok:
                    sonuc.append(kare_cos)
                    # # QLabel boyutlarını ayarla

        return sonuc,diff_matrix_ok
    
    def manhattan_def(self,patches1, patches2, adet, transform, patch_w, patch_h ,manhattan_esik):
        print("Manhattan İçinde")
        diff_scores_mn = []  # Fark skorları listesi
        for p1, p2 in zip(patches1, patches2):  # Yamaları eşleştir
            p1 = transform(Image.fromarray(p1)).unsqueeze(0).to('cuda')   # Tensöre dönüştür ve boyut ekle
            p2 = transform(Image.fromarray(p2)).unsqueeze(0).to('cuda') 
            # Yamaları tensörlere dönüştürüyoruz ve modelin beklediği boyuta getiriyoruz.

            with torch.no_grad():  # Gradyanları hesaplama
                feat1 = model(p1).flatten().cpu().numpy()  # Özellik vektörünü çıkar, önce CPU'ya taşı, sonra NumPy'ye çevir
                feat2 = model(p2).flatten().cpu().numpy()  # Aynı işlemi diğer yama için de yap

            # Manhattan Mesafesi: ||feat1 - feat2||_1
            similarity = np.sum(np.abs(feat1 - feat2))  # Manhattan Mesafesi hesaplama
            diff_scores_mn.append(similarity)  # Uzaklık değerini listeye ekle

        # Fark skorlarını grid boyutunda bir matrise dönüştür
        diff_matrix = np.array(diff_scores_mn).reshape(adet, adet)
        kareler = []  # Listeyi her seferinde sıfırlıyoruz
        print("Manhattan eşiği:", manhattan_esik)

        # Her yama için farkları kontrol et
        for i in range(adet):
            for j in range(adet):
                # if self.vgg19.isChecked():
                if diff_matrix[i, j] > manhattan_esik:  # Eşik değerinden büyükse
                    x1, y1 = j * patch_w, i * patch_h  # Kare koordinatları
                    x2, y2 = (j + 1) * patch_w, (i + 1) * patch_h
                    kareler.append((x1, y1, x2, y2))  # Çizilen kareyi listeye ekle
                # else:
                #     if diff_matrix[i, j] > esik_manhattan:  # Eşik değerinden büyükse
                #         x1, y1 = j * patch_w, i * patch_h  # Kare koordinatları
                #         x2, y2 = (j + 1) * patch_w, (i + 1) * patch_h
                #         kareler.append((x1, y1, x2, y2))  # Çizilen kareyi listeye ekle
        print("MANHATTAN FOR SONRASI")   

        return kareler,diff_matrix

    def oklid_def(self,patches1, patches2, adet, transform, patch_w, patch_h ,oklid_esik):
        print("Öklid İçinde")
        diff_scores_ok = []  # Fark skorları listesi
        for p1, p2 in zip(patches1, patches2):  # Yamaları eşleştir
            p1 = transform(Image.fromarray(p1)).unsqueeze(0).to('cuda')   # Tensöre dönüştür ve boyut ekle
            p2 = transform(Image.fromarray(p2)).unsqueeze(0).to('cuda') 
            # Yamaları tensörlere dönüştürüyoruz ve modelin beklediği boyuta getiriyoruz.

            with torch.no_grad():  # Gradyanları hesaplama
                feat1 = model(p1).flatten().cpu().numpy()  # Özellik vektörünü çıkar, önce CPU'ya taşı, sonra NumPy'ye çevir
                feat2 = model(p2).flatten().cpu().numpy()  # Aynı işlemi diğer yama için de yap

            similarity = np.linalg.norm(feat1 - feat2)
            diff_scores_ok.append(similarity)  # Uzaklık değerini listeye ekle

        # Fark skorlarını grid boyutunda bir matrise dönüştür
        diff_matrix = np.array(diff_scores_ok).reshape(adet, adet)
        kareler_ok = []  # Listeyi her seferinde sıfırlıyoruz

        # Her yama için farkları kontrol et
        for i in range(adet):
            for j in range(adet):
                if diff_matrix[i, j] > oklid_esik:  # Eşik değerinden büyükse
                    x1, y1 = j * patch_w, i * patch_h  # Kare koordinatları
                    x2, y2 = (j + 1) * patch_w, (i + 1) * patch_h
                    kareler_ok.append((x1, y1, x2, y2))  # Çizilen kareyi listeye ekle
        
    
        return kareler_ok,diff_matrix

    def cosinus_def(self,patches1, patches2, adet, transform, patch_w, patch_h ,cos_esik):
        print("Cos İçinde")
        diff_scores_cos = []  # Fark skorları listesi
        for p1, p2 in zip(patches1, patches2):  # Yamaları eşleştir
            p1 = transform(Image.fromarray(p1)).unsqueeze(0).to('cuda')   # Tensöre dönüştür ve boyut ekle
            p2 = transform(Image.fromarray(p2)).unsqueeze(0).to('cuda') 
            # Yamaları tensörlere dönüştürüyoruz ve modelin beklediği boyuta getiriyoruz.

            with torch.no_grad():  # Gradyanları hesaplama
                feat1 = model(p1).flatten().cpu().numpy()  # Özellik vektörünü çıkar, önce CPU'ya taşı, sonra NumPy'ye çevir
                feat2 = model(p2).flatten().cpu().numpy()  # Aynı işlemi diğer yama için de yap

            similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
            diff_scores_cos.append(similarity)  # Uzaklık değerini listeye ekle

        # Fark skorlarını grid boyutunda bir matrise dönüştür
        diff_matrix = np.array(diff_scores_cos).reshape(adet, adet)
        kareler_cos = []  # Listeyi her seferinde sıfırlıyoruz

        # Her yama için farkları kontrol et
        for i in range(adet):
            for j in range(adet):
                if diff_matrix[i, j] < cos_esik:  # Eşik değerinden büyükse
                    x1, y1 = j * patch_w, i * patch_h  # Kare koordinatları
                    x2, y2 = (j + 1) * patch_w, (i + 1) * patch_h
                    kareler_cos.append((x1, y1, x2, y2))  # Çizilen kareyi listeye ekle
        
    
        return kareler_cos,diff_matrix

    def ssim_def(self,img1, img2, ssim_thresh=0.45, min_area=2500):

        blur = 5  # sabit bulanıklık

        g1 = cv2.GaussianBlur(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (blur, blur), 0)
        g2 = cv2.GaussianBlur(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (blur, blur), 0)

        score, full_map = ssim(g1, g2, full=True)
        diff = (1 - full_map) * 255
        diff = diff.astype("uint8")

        _, mask = cv2.threshold(diff, int(ssim_thresh * 255), 255, cv2.THRESH_BINARY)

        valid_mask = (cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) > 5).astype("uint8")
        mask = cv2.bitwise_and(mask, mask, mask=valid_mask)

        # Morfolojik temizlik
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        kareler = []
        vis = img2.copy()

        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            kareler.append((x, y, x + w, y + h))
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return kareler, mask

class Pencere(QMainWindow):
    def __init__(self):
        super().__init__()
        self.fark=FarkWorker()
        uic.loadUi("untitled_2.ui", self)  # arayüzü direkt yükle
        self.setFixedSize(1084, 155)  # Pencere boyutunu sabitle

        self.resim_1_cek.clicked.connect(self.resim_1_ck)
        self.resim_2_cek.clicked.connect(self.resim_2_ck)
        self.cikti.clicked.connect(self.resim_sonuc)
        self.kaydet.clicked.connect(self.kaydet_img)
        self.resnet152.setChecked(True)
        self.vgg19.toggled.connect(self.checkbox_changed)
        self.cosinus.setChecked(True)
        self.hamburger_bar_btn.clicked.connect(self.hamburger_bar_acik)
        
        self.web_btn.clicked.connect(self.web_side)
        self.link_btn.clicked.connect(self.linkedin_link)
        self.git_btn.clicked.connect(self.git_link)

        self.kenar_check.toggled.connect(self.kenar_changed)
        self.gurultu_check.toggled.connect(self.kenar_changed)
        self.keskin_check.toggled.connect(self.kenar_changed)
        self.gri_check.toggled.connect(self.kenar_changed)

        self.ssim.toggled.connect(self.checkbox_changed)

        self.ilk_renk = self.resim_2_cek.styleSheet()

        self.efektler_widget.setVisible(False)

        self.menu_acik = True  # Menü kapalı durum bilgisi

        self.yenilemebool=False
        self.yenileme_calistir=False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.yenileme)


        self.img1=None
        self.img2=None
        self.resim1_path = None
        self.resim2_path = None
        self.masaustu_yolu = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
    
    def ekranda_goster(self,img1,img2):
        rgb13 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        h13, w13, ch13 = rgb13.shape
        qimg1 = QImage(rgb13.data, w13, h13, ch13 * w13, QImage.Format_RGB888)
        self.heatmap_widget.setPixmap(QPixmap.fromImage(qimg1))
        self.heatmap_widget.setScaledContents(True)

        rgb21 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        h21, w21, ch21 = rgb21.shape
        qimg21 = QImage(rgb21.data, w21, h21, ch21 * w21, QImage.Format_RGB888)
        self.cikti_widget.setPixmap(QPixmap.fromImage(qimg21))
        self.cikti_widget.setScaledContents(True)
        self.setFixedSize(1084, 530 )  # Pencere boyutunu sabitl

    def resim_1_ck(self):
        global img1 , img2,file_name_1,file_name_2

        file_name_1, _ = QFileDialog.getOpenFileName(self, "Birinci resmi seç", self.masaustu_yolu, "Resim Dosyaları (*.png *.jpg *.jpeg *.JPG)")
        if file_name_1 and self.img1 is None:
            self.resim1_path = file_name_1

            self.img1=cv2.imread(self.resim1_path)

            self.img1=cv2.resize(self.img1,(1080,720))
            if self.img1 is None:
                QMessageBox.warning(self, "Hata", "Geçersiz bir resim seçtiniz. Lütfen geçerli bir resim seçin.")
                self.resim1_path = None
                return
            self.resim_1_cek.setStyleSheet("background-color: green;")
        elif self.img1 is not None and self.img2 is not None:
            self.setFixedSize(1084, 155)  # Pencere boyutunu sabitle
            self.yenileme_calistir=False
            self.timer.stop()
            self.img1=None
            self.img2=None
            self.resim2_path=None
            self.resim_1_cek.setStyleSheet(self.ilk_renk)
            self.resim_2_cek.setStyleSheet(self.ilk_renk)
            self.resim1_path = file_name_1

            self.img1=cv2.imread(self.resim1_path)
            
            if self.img1 is None:
                QMessageBox.warning(self, "Hata", "Geçersiz bir resim seçtiniz. Lütfen geçerli bir resim seçin.")
                self.resim2_path = None
                return
            
            self.img1=cv2.resize(self.img1,(1080,720))
            self.resim_1_cek.setStyleSheet("background-color: green;")
            self.gecen_sure_degisken.setText("")
            self.islem_label.setText("")
            self.islem_label_2.setText("")

        if self.resim1_path and self.resim2_path:
            self.img1, self.img2 = self.fark.align_and_crop_images(self.img1, self.img2)
            self.ekranda_goster(self.img1,self.img2)
            
    def resim_2_ck(self):
        global img2 , img1,file_name_2


        file_name_2, _ = QFileDialog.getOpenFileName(self, "İkinci resmi seç", self.masaustu_yolu, "Resim Dosyaları (*.png *.jpg *.jpeg *.JPG )")
        if file_name_2 and self.img2 is None:
            self.resim2_path = file_name_2
            self.img2=cv2.imread(self.resim2_path)

            self.img2=cv2.resize(self.img2,(1080,720))
            False
            if self.img2 is None:
                QMessageBox.warning(self, "Hata", "Geçersiz bir resim seçtiniz. Lütfen geçerli bir resim seçin.")
                self.resim2_path = None
                return
            self.resim_2_cek.setStyleSheet("background-color: green;")
        elif self.img1 is not None and self.img2 is not None:
            self.setFixedSize(1084, 155)  # Pencere boyutunu sabitle
            self.yenileme_calistir=False
            self.timer.stop()
            self.img1=None
            self.img2=None
            self.resim1_path=None
            self.resim_1_cek.setStyleSheet(self.ilk_renk)
            self.resim_2_cek.setStyleSheet(self.ilk_renk)
            self.resim2_path = file_name_2
            self.img2=cv2.imread(self.resim2_path)
            if self.img2 is None:
                QMessageBox.warning(self, "Hata", "Geçersiz bir resim seçtiniz. Lütfen geçerli bir resim seçin.")
                self.resim2_path = None
                return
            self.img2=cv2.resize(self.img2,(1080,720))
            self.resim_2_cek.setStyleSheet("background-color: green;")
            self.gecen_sure_degisken.setText("")
            self.islem_label.setText("")
            self.islem_label_2.setText("")




        if self.resim1_path and self.resim2_path:
            self.img1, self.img2 = self.fark.align_and_crop_images(self.img1, self.img2)
            self.ekranda_goster(self.img1,self.img2)

    def checkbox_changed(self):

        if  self.vgg19.isChecked():
            self.parca_adedi.setEnabled(True)

            self.cos_esik.setValue(0.35)
            self.oklid_esik.setValue(80)
            self.manhattan_esik.setEnabled(False)
            self.oklid_and_manhattan.setEnabled(False)
            self.manhattan.setEnabled(False)
            self.oklid.setEnabled(True)
            self.cosinus.setEnabled(True)
            self.cos_esik.setEnabled(True)
            self.oklid_esik.setEnabled(True)
            self.gurultu_check.setEnabled(True)
            self.keskin_check.setEnabled(True)
            self.gri_check.setEnabled(True)
            self.kenar_check.setEnabled(True)
            if self.manhattan.isChecked() or self.oklid_and_manhattan.isChecked():
                self.cosinus.setChecked(True)
            self.kenar_changed()

        elif  self.resnet152.isChecked():
            self.parca_adedi.setEnabled(True)

            self.cos_esik.setValue(0.80)
            self.oklid_esik.setValue(15)
            self.manhattan_esik.setValue(450)
            self.manhattan_esik.setEnabled(True)
            self.oklid_and_manhattan.setEnabled(True)
            self.manhattan.setEnabled(True)
            self.oklid.setEnabled(True)
            self.cosinus.setEnabled(True)  
            self.cos_esik.setEnabled(True)
            self.oklid_esik.setEnabled(True)
            self.gurultu_check.setEnabled(True)
            self.keskin_check.setEnabled(True)
            self.gri_check.setEnabled(True)
            self.kenar_check.setEnabled(True)
            self.kenar_changed()

        elif self.ssim.isChecked():

            self.gurultu_check.setEnabled(False)
            self.keskin_check.setEnabled(False)
            self.gri_check.setEnabled(False)
            self.kenar_check.setEnabled(False)
            self.cosinus.setChecked(False)
            self.manhattan.setChecked(False)
            self.oklid.setChecked(False)
            self.oklid_and_manhattan.setChecked(False)
            self.cos_esik.setEnabled(False)
            self.oklid_esik.setEnabled(False)
            self.manhattan_esik.setEnabled(False)
            self.cosinus.setEnabled(False)
            self.manhattan.setEnabled(False)
            self.oklid.setEnabled(False)
            self.oklid_and_manhattan.setEnabled(False)
            self.parca_adedi.setEnabled(False)

                 
    def model_secim(self):
        global model
        if self.resnet152.isChecked():
            model = models.resnet152(weights="IMAGENET1K_V1")  # ResNet18 modelini ImageNet ağırlıklarıyla yükle
            model = torch.nn.Sequential(*list(model.children())[:-1])  
            model = model.to('cuda')  # Modeli GPU'ya taşıyın
            model.eval()  
            return model
        elif self.vgg19.isChecked():
            model = models.vgg19(weights="IMAGENET1K_V1")  # ResNet18 modelini ImageNet ağırlıklarıyla yükle
            model = torch.nn.Sequential(
                model.features,
                model.avgpool,
            )
            model = model.to('cuda')  # Modeli GPU'ya taşıyın
            model.eval()  
            return model
   
   
    def ana_ekran_kilit(self):
        self.gurultu_check.setEnabled(False)
        self.keskin_check.setEnabled(False)
        self.gri_check.setEnabled(False)
        self.kenar_check.setEnabled(False)
        self.parca_adedi.setEnabled(False)
        self.resnet152.setEnabled(False)
        self.ssim.setEnabled(False)
        self.vgg19.setEnabled(False)
        self.cos_esik.setEnabled(False)
        self.oklid_esik.setEnabled(False)
        self.manhattan_esik.setEnabled(False)
        self.cosinus.setEnabled(False)
        self.manhattan.setEnabled(False)
        self.oklid.setEnabled(False)
        self.oklid_and_manhattan.setEnabled(False)
        self.resim_1_cek.setEnabled(False)
        self.resim_2_cek.setEnabled(False)
        self.kaydet.setEnabled(False)

    def ana_ekran_acik(self):
        self.gurultu_check.setEnabled(True)
        self.keskin_check.setEnabled(True)
        self.gri_check.setEnabled(True)
        self.kenar_check.setEnabled(True)
        self.parca_adedi.setEnabled(True)
        self.resnet152.setEnabled(True)
        self.ssim.setEnabled(True)
        self.vgg19.setEnabled(True)
        self.cos_esik.setEnabled(True)
        self.oklid_esik.setEnabled(True)
        self.manhattan_esik.setEnabled(True)
        self.resim_1_cek.setEnabled(True)
        if  self.vgg19.isChecked():
            
            self.manhattan_esik.setEnabled(False)
            self.oklid_and_manhattan.setEnabled(False)
            self.manhattan.setEnabled(False)
            self.oklid.setEnabled(True)
            self.cosinus.setEnabled(True)

            if self.manhattan.isChecked() or self.oklid_and_manhattan.isChecked():
                self.cosinus.setChecked(True)

        elif  self.resnet152.isChecked():

            self.manhattan_esik.setEnabled(True)
            self.oklid_and_manhattan.setEnabled(True)
            self.manhattan.setEnabled(True)
            self.oklid.setEnabled(True)
            self.cosinus.setEnabled(True)  
        elif self.ssim.isChecked():

            self.gurultu_check.setEnabled(False)
            self.keskin_check.setEnabled(False)
            self.gri_check.setEnabled(False)
            self.kenar_check.setEnabled(False)
            self.cosinus.setChecked(False)
            self.manhattan.setChecked(False)
            self.oklid.setChecked(False)
            self.oklid_and_manhattan.setChecked(False)
            self.cos_esik.setEnabled(False)
            self.oklid_esik.setEnabled(False)
            self.manhattan_esik.setEnabled(False)
            self.cosinus.setEnabled(False)
            self.manhattan.setEnabled(False)
            self.oklid.setEnabled(False)
            self.oklid_and_manhattan.setEnabled(False)
        self.kaydet.setEnabled(True)

        self.resim_2_cek.setEnabled(True)

    
    def resim_sonuc(self):
        print("FARK BUL BUTONUNDA")


        start_time = time.time()
        global model 
        self.yenileme_calistir=False
        self.timer.stop()
        if self.img1 is None or  self.img2 is None:
            if self.img1 is None :
                print("Resimler 1. yüklenmemiş.")
                QMessageBox.warning(self, "Uyarı", "Lütfen  1. resmi de yükleyin!")
            elif self.img2 is None :
                print("Resimler 2. yüklenmemiş.")
                QMessageBox.warning(self, "Uyarı", "Lütfen he 2. resmi de yükleyin!")
            return
       
        self.img2_islenmis = self.img2.copy() 
        self.img1_islenmis=self.img1.copy()


        if self.gri_check.isChecked():
            print("Gri tonlama işlemi yapılıyor")
            gri1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            gri2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
            self.img1_islenmis = cv2.merge([gri1, gri1, gri1])
            self.img2_islenmis = cv2.merge([gri2, gri2, gri2])

                
        
        if self.gurultu_check.isChecked():
                print("Gürültü giderme işlemi yapılıyor")
                self.img1_islenmis = cv2.fastNlMeansDenoisingColored(self.img1_islenmis, None, 10, 10, 7, 21)
                self.img2_islenmis = cv2.fastNlMeansDenoisingColored(self.img2_islenmis, None, 10, 10, 7, 21)

        if self.keskin_check.isChecked():
            print("Keskinleştirme işlemi yapılıyor")
            kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
            self.img1_islenmis = cv2.filter2D(self.img1_islenmis, -1, kernel)
            self.img2_islenmis = cv2.filter2D(self.img2_islenmis, -1, kernel)
        

        if self.kenar_check.isChecked():
            print("Kenar tespiti işlemi yapılıyor")
            self.img1_islenmis = cv2.Canny(self.img1_islenmis, 100, 200)
            self.img2_islenmis = cv2.Canny(self.img2_islenmis, 100, 200)
  # Kenar tespiti sonrası görüntü 1 kanallı olur, tekrar 3 kanallı yapmak için:
            self.img1_islenmis = cv2.cvtColor(self.img1_islenmis, cv2.COLOR_GRAY2BGR)  # 3 kanallı hale getir
            self.img2_islenmis = cv2.cvtColor(self.img2_islenmis, cv2.COLOR_GRAY2BGR)  # 3 kanallı hale getir


        if not self.resnet152.isChecked() and not self.vgg19.isChecked() and not self.ssim.isChecked():
            print("MODEL SEÇİLMEMİŞ")
            QMessageBox.warning(self, "Uyarı", "Lütfen bir model seçiniz!")
            return 
        self.ekranda_goster(self.img1_islenmis,self.img2_islenmis)

        self.gecen_sure_degisken.setText("")
        self.islem_label.setText("")

        self.islem_label_2.setText("FARK İŞLEMİ UYGULANMAKTA")
        self.islem_label_2.setStyleSheet("color: red; ")
        self.ana_ekran_kilit()
        QApplication.processEvents()  # UI hemen güncellensin
        model = self.model_secim()  # model_secim fonksiyonunu çağırarak model seçimi yap
        self.adet = self.parca_adedi.value()
        patch_h, patch_w,patches1 , patches2 = self.fark.get_patches_for_both_images(self.img1_islenmis,self.img2_islenmis,self.adet)
        transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Yeniden boyutlandır
        transforms.ToTensor(),  # Tensöre dönüştür
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
           ])
        print("FARK BUL BUTONUNDA")
        self.ssm = None
        if self.ssim.isChecked():
            kareler, self.ssm = self.fark.ssim_def(self.img1, self.img2)
        elif self.oklid_and_manhattan.isChecked():
            self.esik_manhattan1=self.manhattan_esik.value()
            self.esik_oklid1=self.oklid_esik.value()
            kareler,diff_matrix=self.fark.oklid_and_manhattan_def(patches1,patches2,self.adet,transform,patch_w,patch_h,self.esik_manhattan1,self.esik_oklid1)
        elif self.manhattan.isChecked():
            self.esik_manhattan1=self.manhattan_esik.value()
            kareler,diff_matrix=self.fark.manhattan_def(patches1, patches2, self.adet, transform, patch_w, patch_h,self.esik_manhattan1)
        elif self.oklid.isChecked():
            self.esik_oklid1=self.oklid_esik.value()
            kareler,diff_matrix=self.fark.oklid_def(patches1, patches2, self.adet, transform, patch_w, patch_h,self.esik_oklid1) 
        elif self.cosinus.isChecked():
           self.cos_esik1=self.cos_esik.value()
           kareler ,diff_matrix=  self.fark.cosinus_def(patches1, patches2, self.adet, transform, patch_w, patch_h,self.cos_esik1)

        else:
            print("FARK İŞLEMİ SEÇİLMEMİŞ")
            QMessageBox.warning(self, "Uyarı", "Lütfen bir işlem seçiniz!")
            return        
        self.setFixedSize(1084, 530 )  # Pencere boyutunu sabitl

        for (x1, y1, x2, y2) in kareler:
            if self.ssim.isChecked():
                print("Farklar eşleşti, kare çiziliyor!")
                cv2.rectangle(self.img2_islenmis, (x1, y1), (x2, y2), (0, 0, 225), 2) 
                cv2.rectangle(self.img1_islenmis, (x1, y1), (x2, y2), (0, 0, 225), 2)
            else:
                print("Farklar eşleşti, kare çiziliyor!")
                cv2.rectangle(self.img2_islenmis, (x1, y1), (x2, y2), (0, 0, 225), 2) 
                cv2.rectangle(self.img1_islenmis, (x1, y1), (x2, y2), (0, 0, 225), 2)
        self.timer.start(1000)  

        if self.yenileme_calistir is False:
            self.yenileme_calistir=True
            rgb1 = cv2.cvtColor(self.img2_islenmis, cv2.COLOR_BGR2RGB)
            h1, w1, ch1 = rgb1.shape
            self.img2_islenmis= QImage(rgb1.data, w1, h1, ch1 * w1, QImage.Format_RGB888)

            rgb12 = cv2.cvtColor(self.img1_islenmis, cv2.COLOR_BGR2RGB)
            h12, w12, ch12 = rgb12.shape
            self.img1_islenmis= QImage(rgb12.data, w12, h12, ch12 * w12, QImage.Format_RGB888) 


        if self.ssim.isChecked():
            rgb5 = cv2.cvtColor(self.ssm, cv2.COLOR_BGR2RGB)
            h5, w5, ch5 = rgb5.shape
            self.ssm= QImage(rgb5.data, w5, h5, ch5 * w5, QImage.Format_RGB888)
            self.ssm
            self.heatmap_widget.setPixmap(QPixmap.fromImage(self.ssm))

        else:
            widget_width = self.heatmap_widget.width()
            widget_height = self.heatmap_widget.height()
            dpi = 50  # Matplotlib default DPI
            fig = plt.figure(figsize=(widget_width / dpi, widget_height / dpi), dpi=dpi)
            
            self.heatmap_widget.setScaledContents(True)
            self.heatmap_widget.setAlignment(Qt.AlignCenter)
            if self.cosinus.isChecked():
                ax = sns.heatmap(diff_matrix, cmap="RdYlBu", annot=False, cbar=True, xticklabels=False, yticklabels=False)
            else:
                ax = sns.heatmap(diff_matrix, cmap="RdYlBu_r", annot=False, cbar=True, xticklabels=False, yticklabels=False)

            ax.axis('on')  # Eksen çizgileri
            ax.set_title('Heatmap ', fontsize=16)

            ax.set_xticks([])  # X ekseni yazıları
            ax.set_yticks([])
            ax.set_frame_on(False)  # Çerçeveyi kapat

            fig.patch.set_visible(False)  # Figür etrafındaki kenarlıkları kaldırır
# Açıklayıcı alt yazı
            fig.text(0.5, 0.02, "Kırmızılığın arttığı yerde fark artar", ha='center', fontsize=15, color='red')


            # Figure'ı render et
            canvas = FigureCanvasAgg(fig)
            canvas.draw()

            # RGBA piksel verisini ve boyutları al
            buf, (width, height) = canvas.print_to_buffer()

            # QImage oluştur (RGBA8888 formatı)
            image = QImage(buf, width, height, QImage.Format_RGBA8888)

            # QPixmap'e çevir ve widget'e aktar
            pixmap = QPixmap.fromImage(image)
            self.heatmap_widget.setPixmap(pixmap)

            plt.close(fig)

            # QImage'den NumPy array'e çevir
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            arr = np.array(ptr).reshape((image.height(), image.width(), 4))  # RGBA

            # RGBA'yı BGR formatına dönüştür (OpenCV uyumlu hale getir)
            self.heatmap = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)


        finish = time.time()
        elapsed = finish - start_time
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        self.gecen_sure_degisken.setText(f"{minutes} dakika {seconds:.1f} saniye")
        self.gecen_sure_degisken.setStyleSheet("color: green;")

        self.islem_label.setText("Geçen Süre")
        self.islem_label.setStyleSheet("color: green; font-size: 18pt;")

        self.islem_label_2.setText("İşlem Tamamlandı")
        self.islem_label_2.setStyleSheet("color: red; ")
        self.ana_ekran_acik()
        QApplication.processEvents()  # UI hemen güncellensin
        self.kaydet_degerleri()


    def yenileme(self):
        if self.img1_islenmis is not None and self.img2_islenmis is not None and self.yenileme_calistir:
            if self.yenilemebool is True :

                self.cikti_widget.setPixmap(QPixmap.fromImage(self.img2_islenmis))
                self.cikti_widget.setScaledContents(True)
                self.yenilemebool=False
            else:

                self.cikti_widget.setPixmap(QPixmap.fromImage(self.img1_islenmis))
                self.cikti_widget.setScaledContents(True)
                self.yenilemebool=True


    def kaydet_img(self):
        def qimage_to_np(qimg):
            qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
            width = qimg.width()
            height = qimg.height()
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            arr = np.array(ptr).reshape((height, width, 3))
            return arr

        # Verilerin varlığını kontrol et
        if not hasattr(self, "img1_islenmis") or self.img1_islenmis is None:
            QMessageBox.warning(self, "Uyarı", "İşlenmiş birinci görüntü yok!")
            return
        if not hasattr(self, "img2_islenmis") or self.img2_islenmis is None:
            QMessageBox.warning(self, "Uyarı", "İşlenmiş ikinci görüntü yok!")
            return
        if self.ssm is None and  self.ssim.isChecked():
            QMessageBox.warning(self, "Uyarı", "SSIM Haritası bulunamadı!")
            return
        
        elif (not hasattr(self, "heatmap") or self.heatmap is None) and not self.ssim.isChecked():
            QMessageBox.warning(self, "Uyarı", "Heatmap bulunamadı!")
            return


        # QImage ise NumPy'ye çevir
        if isinstance(self.img1_islenmis, QImage):
            self.img1_islenmis2 = qimage_to_np(self.img1_islenmis)
        else:
            self.img1_islenmis2=self.img1_islenmis.copy()
        if isinstance(self.img2_islenmis, QImage):
            self.img2_islenmis2 = qimage_to_np(self.img2_islenmis)
        else:
            self.img2_islenmis2=self.img2_islenmis.copy()
        if not self.ssim.isChecked():
            if isinstance(self.heatmap, QImage) :
                self.heatmap2 = qimage_to_np(self.heatmap)
            else:
                self.heatmap2=self.heatmap.copy()
        # Tüm görüntülerin aynı yüksekliğe getirilmesi
            h = min(self.img1_islenmis2.shape[0], self.img2_islenmis2.shape[0])
            w = self.heatmap2.shape[1]

            img1_np = cv2.resize(self.img1_islenmis2, (w, h))
            img2_np = cv2.resize(self.img2_islenmis2, (w, h))
            heatmap_resized = cv2.resize(self.heatmap2, (w, h))
        else:
            if isinstance(self.ssm, QImage) :
                self.ssm2 = qimage_to_np(self.ssm)
            else:
                self.ssm2=self.ssm.copy()
            h = min(self.img1_islenmis2.shape[0], self.img2_islenmis2.shape[0])
            w = self.ssm2.shape[1]

            img1_np = cv2.resize(self.img1_islenmis2, (w, h))
            img2_np = cv2.resize(self.img2_islenmis2, (w, h))
            heatmap_resized = cv2.resize(self.ssm2, (w, h))
        # Beyaz boşluk (20 piksel genişliğinde)
        padding = np.zeros((h, 20, 3), dtype=np.uint8)

        # Görselleri birleştir
        combined = np.hstack((padding,img1_np, padding, img2_np, padding, heatmap_resized,padding))
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        # Beyaz boşluk (üst kısım için yazı alanı)
        combined=cv2.resize(combined,(1280,720))
        padding_top = np.ones((40, combined.shape[1], 3), dtype=np.uint8) * 255  # Beyaz üst bant
        
        # Yazı ekle
        font = cv2.FONT_HERSHEY_COMPLEX
        text = ' '.join(self.ekran_yazisi)
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = (padding_top.shape[1] - text_size[0]) // 2
        text_y = (padding_top.shape[0] + text_size[1]) // 2

        cv2.putText(padding_top, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        # Üst yazı alanı + içerik birleştirme
        final_image = np.vstack((padding_top, combined))

        dosya_yolu, _ = QFileDialog.getSaveFileName(
            self,
            "Görseli Kaydet",
            f"{self.masaustu_yolu}/fark_çıktısı",
            "PNG Dosyası (*.png);;JPEG Dosyası (*.jpg);;Tüm Dosyalar (*)"
        )

        if dosya_yolu:
            # Uzantı yoksa .png ekle
            if not (dosya_yolu.lower().endswith(".png") or dosya_yolu.lower().endswith(".jpg") or dosya_yolu.lower().endswith(".jpeg")):
                dosya_yolu += ".png"


        if dosya_yolu:
            final_image=cv2.resize(final_image,(1280,720))
            cv2.imwrite(dosya_yolu, final_image)
            print("Görsel kaydedildi:", dosya_yolu)



    def hamburger_bar_acik(self):
        print("hamburger buton içindde")
        if self.menu_acik:
            print("hamburger if içi")
            self.efektler_widget.setVisible(True)

            self.menu_acik = False

        else:
            print("hamburger else içi")

            self.efektler_widget.setVisible(False)
            self.menu_acik = True

    def kenar_changed(self):
        if self.kenar_check.isChecked():
            self.gri_check.setChecked(False)
            self.gurultu_check.setChecked(False)
            self.keskin_check.setChecked(False)
            
            self.gri_check.setEnabled(False)
            self.gurultu_check.setEnabled(False)
            self.keskin_check.setEnabled(False)
        else:
            # Kenar tespiti kaldırıldığında yapılacaklar
            self.keskin_check.setEnabled(True)
            self.gri_check.setEnabled(True)
            self.gurultu_check.setEnabled(True)

        if self.gri_check.isChecked() or self.gurultu_check.isChecked() or self.keskin_check.isChecked():
            self.kenar_check.setChecked(False)      
            self.kenar_check.setEnabled(False)
        else:
            self.kenar_check.setEnabled(True)


    def kaydet_degerleri(self):
        self.ekran_yazisi = ['Efektler:']
        if self.gri_check.isChecked():
            self.ekran_yazisi.append('Grilestirme,')
        if self.gurultu_check.isChecked():
            self.ekran_yazisi.append('Gurultu Azaltma,')
        if self.keskin_check.isChecked():
            self.ekran_yazisi.append('Keskinlestirme')
        if self.kenar_check.isChecked():
            self.ekran_yazisi.append('Kenar Algilama')
        if not (self.gri_check.isChecked() or self.gurultu_check.isChecked() or self.keskin_check.isChecked() or self.kenar_check.isChecked()):
            self.ekran_yazisi.append(' --Efekt Uygulanmadi-- ')

        if self.ssim.isChecked():
            self.ekran_yazisi.append('/ SSIM ile fark bulundu')
        elif self.oklid_and_manhattan.isChecked():
            self.ekran_yazisi.append('/ Oklid ve Manhattan ile fark bulundu')
            self.ekran_yazisi.append(f'Esik deger Manhattan: {self.esik_manhattan1} / Oklid: {self.esik_oklid1:.2f}')
        elif self.manhattan.isChecked():
            self.ekran_yazisi.append('/ Manhattan ile fark bulundu')
            self.ekran_yazisi.append(f'Esik deger: {self.esik_manhattan1}')
        elif self.oklid.isChecked():
            self.ekran_yazisi.append('/ Oklid ile fark bulundu')
            self.ekran_yazisi.append(f'Esik deger: {self.esik_oklid1:.2f}')
        elif self.cosinus.isChecked():
            self.ekran_yazisi.append('/ Cosinus ile fark bulundu')
            self.ekran_yazisi.append(f'Esik deger: {self.cos_esik1:.2f}')

    def web_side(self):
            url = QUrl("http://www.yusufkaracor.com.tr/")
            QDesktopServices.openUrl(url)
    def linkedin_link(self):
            url = QUrl("https://tr.linkedin.com/in/yusuf-kara%C3%A7or-7106ysf7106")
            QDesktopServices.openUrl(url)
    def git_link(self):
            url = QUrl("https://tr.linkedin.com/in/yusuf-kara%C3%A7or-7106ysf7106")
            QDesktopServices.openUrl(url)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    pencere = Pencere() 

    pencere.show()

    ekran = app.primaryScreen().availableGeometry()
    pencere.move(ekran.center() - pencere.rect().center())
    sys.exit(app.exec_())

