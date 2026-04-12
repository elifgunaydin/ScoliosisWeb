# train.py
import torch
import torch.nn as nn
import numpy as np
import cv2
import io

# --- MODÜLER YAPI 
# Preprocessing.py dosyasından fonksiyonları çekiyoruz
from preprocessing import load_and_enhance_image, prepare_for_model, encode_result_image, TARGET_SIZE

MODEL_PATH = "multitask_unet_pc_latest.pth" 
NUM_KEYPOINTS = 68 
DEVICE = "cpu"

# --- 1. MODEL MİMARİSİ ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class MultiTaskUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_keypoints=68):
        super().__init__()
        self.d1 = DoubleConv(in_channels, 64); self.pool1 = nn.MaxPool2d(2, 2)
        self.d2 = DoubleConv(64, 128); self.pool2 = nn.MaxPool2d(2, 2)
        self.d3 = DoubleConv(128, 256); self.pool3 = nn.MaxPool2d(2, 2)
        self.d4 = DoubleConv(256, 512); self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2); self.u1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2); self.u2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2); self.u3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2); self.u4 = DoubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_landmarks = nn.Sequential(
            nn.Flatten(), nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, n_keypoints * 2)
        )
    def forward(self, x):
        x1 = self.d1(x); p1 = self.pool1(x1)
        x2 = self.d2(p1); p2 = self.pool2(x2)
        x3 = self.d3(p2); p3 = self.pool3(x3)
        x4 = self.d4(p3); p4 = self.pool4(x4)
        b = self.bottleneck(p4)
        d1 = self.u1(torch.cat([x4, self.up1(b)], dim=1))
        d2 = self.u2(torch.cat([x3, self.up2(d1)], dim=1))
        d3 = self.u3(torch.cat([x2, self.up3(d2)], dim=1))
        d4 = self.u4(torch.cat([x1, self.up4(d3)], dim=1))
        return self.final_conv(d4), self.fc_landmarks(self.avg_pool(b))

# --- MODELİ YÜKLE ---
print("Yapay Zeka Modeli Yükleniyor...")
model = MultiTaskUNet(n_keypoints=NUM_KEYPOINTS).to(DEVICE)
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(" Model Başarıyla Yüklendi!")
except Exception as e:
    print(f" Model Yükleme Hatası: {e}")

# --- MATEMATİKSEL HESAPLAMA 
import math
def calculate_cobb_from_mask_contours(mask_numpy, orig_h, orig_w):
    """
    Maskedeki omurları bulur, eğimlerini ölçer ve 
    en eğik olanların çizim koordinatlarını döndürür.
    """
    mask_uint8 = (mask_numpy * 255).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vertebrae_data = [] # (Açı, Merkez, KutuKöşeleri) saklayacağız
    vis_boxes = [] 
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200: continue 

        rect = cv2.minAreaRect(cnt)
        (center, (w, h), angle) = rect
        
        # Dikey/Yatay düzeltmesi
        if w < h:
            tilt_angle = angle
        else:
            tilt_angle = angle + 90
            
        # OpenCV minAreaRect bazen negatif açılar verebilir, normalize edelim
        # Ancak Cobb için relatif fark önemli, şimdilik böyle bırakalım.
        
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        vis_boxes.append(box)
        
        vertebrae_data.append({
            'angle': tilt_angle,
            'center': center,
            'box': box
        })

    if not vertebrae_data: return 0.0, None, []

    # En yüksek ve en düşük açılı omurları bul (Cobb uç noktaları)
    # Not: Lambda fonksiyonu sözlükteki 'angle' değerine göre sıralar
    max_v = max(vertebrae_data, key=lambda x: x['angle'])
    min_v = min(vertebrae_data, key=lambda x: x['angle'])
    
    cobb_angle = max_v['angle'] - min_v['angle']
    
    # Çizim verisi hazırlayalım (Merkezden eğim yönüne uzun çizgi)
    def get_line_coords(center, angle_deg, length=400):
        cx, cy = center
        angle_rad = np.radians(angle_deg)
        # Eğim formülü: y = mx. 
        # OpenCV koordinat sisteminde y aşağı doğru artar.
        # Açıyı dikkate alarak uç noktaları bulalım (dikeyimsi çizgiler için)
        
        # Eğim açısı genellikle yatayla yapılan açıdır ama minAreaRect dikey verebilir.
        # Basit trigonometri ile uzatalım:
        dx = length * np.cos(angle_rad) # Aslında sin/cos minAreaRect formatına göre değişir
        dy = length * np.sin(angle_rad) 
        
        # Eğer omurga dikse (angle ~ 0 veya 90), çizgiyi yatay çekmemiz lazım (endplate paralel)
        # Cobb çizgisi omurga eksenine DİK olur (endplate hizası).
        # Bulduğumuz 'tilt_angle' omurganın kendi duruş açısı.
        # Çizeceğimiz çizgi buna 90 derece dik olmalı.
        perp_angle_rad = np.radians(angle_deg + 90) # 90 derece ekle
        
        dx = length * np.cos(perp_angle_rad)
        dy = length * np.sin(perp_angle_rad)
        
        p1 = (int(cx - dx), int(cy - dy))
        p2 = (int(cx + dx), int(cy + dy))
        return p1, p2

    vis_lines = {
        'top_line': get_line_coords(max_v['center'], max_v['angle']),
        'bottom_line': get_line_coords(min_v['center'], min_v['angle'])
    }
    
    return abs(cobb_angle), vis_lines, vis_boxes
# --- 3. ANA FONKSİYON ---
# ai_engine.py dosyasında analyze_spine_image fonksiyonunu bul ve bu kısımları değiştir:

def analyze_spine_image(file_stream):
    # 1. YÜKLEME
    img_gray, orig_h, orig_w = load_and_enhance_image(file_stream)
    
    # Web önizleme
    img_display_512 = cv2.resize(img_gray, (TARGET_SIZE, TARGET_SIZE))
    preprocessed_base64 = encode_result_image(img_display_512)
    
    # 2. TAHMİN
    tensor_img = prepare_for_model(img_gray)
    tensor_img = tensor_img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_mask, _ = model(tensor_img)
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = (pred_mask > 0.5).float().squeeze().cpu().numpy()

    # 3. MASKE BÜYÜTME
    full_size_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # --- YENİ EKLENEN KISIM: SADECE MASKE GÖRÜNTÜSÜ OLUŞTURMA ---
    # Maskeyi 0-1 aralığından 0-255 aralığına çekiyoruz (Siyah-Beyaz resim için)
    mask_visual = (full_size_mask * 255).astype(np.uint8)
    # İsterseniz maskeyi yeşil veya farklı renk yapmak için renklendirebilirsiniz, şimdilik beyaz omurga-siyah arka plan yapıyoruz:
    mask_base64 = encode_result_image(mask_visual)
    # -------------------------------------------------------------

    # 4. HESAPLAMA (angle, lines, boxes)
    angle, lines, boxes = calculate_cobb_from_mask_contours(full_size_mask, orig_h, orig_w)

    # 5. GÖRSELLEŞTİRME (Mevcut Kodlar Aynen Kalıyor)
    result_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # ... (Overlay, box çizimleri, çizgiler buradaki mevcut kodlarınız...) ...
    # Mevcut kodlarınızdaki görselleştirme işlemleri...
    overlay = result_img.copy()
    overlay[full_size_mask == 1] = (0, 255, 0)
    cv2.addWeighted(overlay, 0.3, result_img, 0.7, 0, result_img)
    
    for box in boxes:
        cv2.drawContours(result_img, [box], 0, (0, 255, 255), 1)

    if lines:
        p1_t, p2_t = lines['top_line']
        cv2.line(result_img, p1_t, p2_t, (0, 0, 255), 4)
        p1_b, p2_b = lines['bottom_line']
        cv2.line(result_img, p1_b, p2_b, (255, 0, 0), 4)

    final_base64 = encode_result_image(result_img)
    
    # DÖNÜŞ DEĞERİNE mask_base64 EKLENDİ
    return final_base64, preprocessed_base64, f"{angle:.2f}", mask_base64