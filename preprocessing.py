# preprocessing.py
import cv2
import numpy as np
import base64
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.exposure import equalize_adapthist
from skimage.util import img_as_ubyte

# EXIF düzeltmesi için eklenen kütüphaneler
from PIL import Image, ImageOps
import io

# --- ESKİ KONFİGÜRASYONLAR (AYNEN DURUYOR) ---
TARGET_SIZE = 512 
CLAHE_CLIP = 0.01 
BILATERAL_SIGMA = 10 

# --- ESKİ GÖRÜNTÜ İYİLEŞTİRME FONKSİYONLARI (AYNEN DURUYOR) ---
def apply_clahe(img):
    """CLAHE uygular"""
    return img_as_ubyte(equalize_adapthist(img / 255.0, clip_limit=CLAHE_CLIP))

def apply_bilateral_filter(img):
    """Kenarları koruyarak gürültü temizleme uygular"""
    return cv2.bilateralFilter(img, d=5, sigmaColor=BILATERAL_SIGMA, sigmaSpace=BILATERAL_SIGMA)

# --- GÜNCELLENEN ANA YÜKLEME FONKSİYONU ---
def load_and_enhance_image(file_stream):
    file_stream.seek(0)
    
    # 1. YENİ KISIM: Hem web hem mobil için güvenli okuma ve EXIF düzeltme
    try:
        pil_image = Image.open(file_stream)
        # Web'den gelene dokunmaz, Mobilden gelen yan dönmeyi düzeltir:
        pil_image = ImageOps.exif_transpose(pil_image) 
        
        pil_image = pil_image.convert('RGB')
        open_cv_image = np.array(pil_image)
        # OpenCV'nin renk formatına (BGR) çevirir
        img = open_cv_image[:, :, ::-1].copy() 
    except Exception as e:
        raise ValueError(f"Görüntü okunamadı veya dönüştürülemedi: {e}")
    
    if img is None: raise ValueError("Görüntü okunamadı.")
    
    # 2. ESKİ KISIM: Gri tona çevirme ve yapay zeka ön işleme (Aynen duruyor)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    orig_h, orig_w = img_gray.shape

    img_enhanced = apply_bilateral_filter(apply_clahe(img_gray))
    
    return img_enhanced, orig_h, orig_w

# --- ESKİ MODEL HAZIRLIĞI (AYNEN DURUYOR) ---
def prepare_for_model(img_gray):
    """
    Görüntüyü AI modelinin giriş formatına (512x512, Normalize, Tensor) dönüştürür.
    """
    transform = A.Compose([
        A.Resize(TARGET_SIZE, TARGET_SIZE), 
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return transform(image=img_gray)["image"]

def encode_result_image(result_img_numpy):
    """NumPy dizisini HTML'de göstermek için Base64 formatına kodlar."""
    _, buffer = cv2.imencode('.jpg', result_img_numpy)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"