# test.py mae, rmse, std sapma, dice skoru hesaplama
import os
import statistics
import math
import numpy as np
import cv2
from ai_engine import analyze_spine_image

# --- AYARLAR ---
TEST_IMAGE_FOLDER = r"D:\dataset_final\dataset_final\images\test"
LABEL_FILE = r"D:\spine-project\dataset5000\Cobb_merged.txt"
KABUL_EDILEBILIR_HATA = 5.0 

def dice_score(pred, gt):
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)

    if union == 0:
        return 1.0

    return 2.0 * intersection / union

def load_ground_truth(file_path):
    data = {}
    print(f"'{file_path}' dosyası taranıyor...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(',')
                if len(parts) >= 2:
                    filename = parts[0].strip()
                    try:
                        angles = [float(x) for x in parts[1:]]
                        data[filename] = max(angles)
                    except ValueError:
                        continue
        print(f"Referans veri yüklendi. Toplam kayıt: {len(data)}")
    except FileNotFoundError:
        print(f"HATA: {file_path} dosyası bulunamadı!")
    return data

def run_evaluation():
    print("\n--- DETAYLI DOĞRULUK TESTİ BAŞLIYOR ---\n")
    print(f"Hedef Klasör: {TEST_IMAGE_FOLDER}")
    
    ground_truth = load_ground_truth(LABEL_FILE)
    if not ground_truth: return

    errors = []
    squared_errors = []
    dice_scores = []

    basarili_sayisi = 0
    toplam_islenen = 0

    print(f"{'Dosya Adı':<15} | {'Gerçek':<10} | {'Tahmin':<10} | {'Fark':<8} | {'Durum'}")
    print("-" * 75)

    for filename, real_angle in ground_truth.items():
        file_path = os.path.join(TEST_IMAGE_FOLDER, filename)
        mask_path = img_path.replace("images", "masks").replace(".jpg", ".png")

         if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        
        try:
            with open(img_path, 'rb') as f:
                result = analyze_spine_image(f)

            predicted_angle = float(result["cobb_angle"])
            pred_mask = result["mask_pred"]

            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = (gt_mask > 127).astype(np.uint8)

            dice = dice_score(pred_mask, gt_mask)
            dice_scores.append(dice)

            diff = abs(real_angle - predicted_angle)
            errors.append(diff)
            squared_errors.append(diff ** 2)

            toplam_islenen += 1

            if diff <= KABUL_EDILEBILIR_HATA:
                basarili_sayisi += 1

        except Exception as e:
            print(f"{filename} hata verdi: {e}")

mae = statistics.mean(errors)
    rmse = math.sqrt(statistics.mean(squared_errors))
    std_dev = statistics.stdev(errors) if len(errors) > 1 else 0
    accuracy_rate = (basarili_sayisi / toplam_islenen) * 100
    dice_mean = statistics.mean(dice_scores)

    print("\n" + "="*50)
    print("GENEL PERFORMANS ÖZETİ")
    print("="*50)
    print(f"Test Görüntü Sayısı      : {toplam_islenen}")
    print(f"MAE                     : {mae:.2f}°")
    print(f"RMSE                    : {rmse:.2f}")
    print(f"Standart Sapma          : {std_dev:.2f}")
    print(f"Dice Skoru (Ortalama)   : {dice_mean:.4f}")
    print(f"±{KABUL_EDILEBILIR_HATA}° Başarı Oranı : %{accuracy_rate:.2f}")

if __name__ == "__main__":
    run_evaluation()