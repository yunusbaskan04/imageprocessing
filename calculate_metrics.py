import cv2
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_metrics_for_directory(orig_dir, target_dir):
    """
    İki klasördeki aynı isimli resimleri eşleştirir ve ortalama PSNR ile SSIM değerlerini döndürür.
    """
    psnr_list = []
    ssim_list = []
    
    # Desteklenen formatlar
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Orijinal klasördeki resimleri listele
    image_files = [f for f in os.listdir(orig_dir) if f.lower().endswith(valid_extensions)]
    
    for filename in image_files:
        orig_path = os.path.join(orig_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        # Eğer hedefte resim yoksa atla
        if not os.path.exists(target_path):
            continue
            
        # Görüntüleri oku
        orig_img = cv2.imread(orig_path)
        target_img = cv2.imread(target_path)
        
        if orig_img is None or target_img is None:
            continue
            
        # Olası bir boyut uyuşmazlığına karşı güvenlik (Bozulma işlemi boyut değiştirmiyor ama garanti olsun)
        if orig_img.shape != target_img.shape:
            target_img = cv2.resize(target_img, (orig_img.shape[1], orig_img.shape[0]))
            
        # PSNR Hesaplama
        p_val = psnr(orig_img, target_img, data_range=255)
        
        # SSIM Hesaplama (Renkli görüntüler için channel_axis=-1 HAYATİ öneme sahip)
        s_val = ssim(orig_img, target_img, channel_axis=-1, data_range=255)
        
        psnr_list.append(p_val)
        ssim_list.append(s_val)
        
    # Eğer klasör boşsa veya okunabilen resim yoksa 0 döndür
    if not psnr_list:
        return 0.0, 0.0
        
    return np.mean(psnr_list), np.mean(ssim_list)

def main():
    # Klasör yolları
    orig_dir = os.path.join("dataset_original", "test", "images")
    degraded_base_dir = "dataset_degraded"
    enhanced_base_dir = "dataset_enhanced"
    
    # Kontrol mekanizması
    if not os.path.exists(orig_dir):
        print(f"HATA: Orijinal görüntüler klasörü bulunamadı: {orig_dir}")
        return
        
    subdirs = ["low_light", "fog", "noise"]
    results = {}
    
    print("Metrikler (PSNR & SSIM) hesaplanıyor, lütfen bekleyin...\n")
    
    # Her bozulma türü için iterasyon
    for subdir in subdirs:
        degraded_dir = os.path.join(degraded_base_dir, subdir)
        enhanced_dir = os.path.join(enhanced_base_dir, subdir)
        
        # 1. Orijinal vs Degraded
        deg_psnr, deg_ssim = calculate_metrics_for_directory(orig_dir, degraded_dir)
        
        # 2. Orijinal vs Enhanced
        enh_psnr, enh_ssim = calculate_metrics_for_directory(orig_dir, enhanced_dir)
        
        # Sonuçları kaydet
        results[subdir] = {
            'deg_psnr': deg_psnr,
            'deg_ssim': deg_ssim,
            'enh_psnr': enh_psnr,
            'enh_ssim': enh_ssim
        }
        
    # === TABLOYU YAZDIRMA KISMI ===
    
    print("=" * 80)
    print(" " * 25 + "GÖRÜNTÜ KALİTE METRİKLERİ (PSNR & SSIM)")
    print("=" * 80)
    
    # Sütun başlıkları
    print(f"| {'Bozulma Türü':<13} | {'Bozulmuş (Degraded) Değerler':<27} | {'İyileştirilmiş (Enhanced) Değerler':<29} |")
    print(f"| {'':<13} | {'PSNR (dB)':<12} | {'SSIM':<12} | {'PSNR (dB)':<12} | {'SSIM':<14} |")
    print("-" * 80)
    
    # Verileri yazdırma
    for subdir in subdirs:
        r = results[subdir]
        
        # Sayıları string formatına sokma (PSNR için 2, SSIM için 4 ondalık)
        deg_psnr_str = f"{r['deg_psnr']:.2f}"
        deg_ssim_str = f"{r['deg_ssim']:.4f}"
        
        enh_psnr_str = f"{r['enh_psnr']:.2f}"
        enh_ssim_str = f"{r['enh_ssim']:.4f}"
        
        # İsimleri makale formatına uygun gösterme
        name_map = {"low_light": "Low Light", "fog": "Fog", "noise": "Noise"}
        display_name = name_map.get(subdir, subdir)
        
        print(f"| {display_name:<13} | {deg_psnr_str:<12} | {deg_ssim_str:<12} | {enh_psnr_str:<12} | {enh_ssim_str:<14} |")
        
    print("=" * 80)
    print("Not: PSNR değerinin artması ve SSIM değerinin 1'e yaklaşması görüntü kalitesinin arttığını gösterir.")

if __name__ == "__main__":
    main()
