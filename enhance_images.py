import cv2
import numpy as np
import os

def enhance_image(img):
    """
    Belirlenen parametrelerle sırasıyla: 
    1. BGR -> LAB dönüşümü
    2. L kanalına CLAHE
    3. LAB -> BGR dönüşümü
    4. Bilateral Filter
    işlemlerini uygular. (Ablation study için Morphological closing çıkarıldı)
    """
    # 1. Renk uzayını BGR'den LAB'a çevir
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Kanalları ayır (Luminance, A, B)
    l, a, b = cv2.split(lab_img)
    
    # 2. Sadece L (Luminance) kanalına CLAHE uygula
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # CLAHE uygulanmış L kanalını diğerleriyle birleştir
    merged_lab = cv2.merge((cl, a, b))
    
    # 3. Görüntüyü tekrar LAB'dan BGR'ye çevir
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    # 4. Kenarları korurken gürültüyü silmek için Bilateral Filter
    filtered_img = cv2.bilateralFilter(enhanced_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    
    return filtered_img

def main():
    input_base_dir = "dataset_degraded"
    output_base_dir = "dataset_enhanced"
    
    # Girdi ana klasörünün varlığını kontrol et
    if not os.path.exists(input_base_dir):
        print(f"HATA: Girdi klasörü bulunamadı: {input_base_dir}")
        print("Lütfen önce bozulma scriptini (generate_degradations.py) çalıştırdığınızdan emin olun.")
        return
        
    # dataset_degraded içindeki alt klasörleri dolaşacağız
    subdirs = ["low_light", "fog", "noise"]
    
    for subdir in subdirs:
        input_dir = os.path.join(input_base_dir, subdir)
        output_dir = os.path.join(output_base_dir, subdir)
        
        # Eğer bu alt klasör yoksa atla
        if not os.path.exists(input_dir):
            print(f"UYARI: Alt klasör bulunamadı, atlanıyor: {input_dir}")
            continue
            
        # Çıktı alt klasörünü güvenli bir şekilde oluştur
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"HATA: {output_dir} klasörü oluşturulamadı: {e}")
            continue
            
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
        
        if not image_files:
            print(f"UYARI: {input_dir} içerisinde resim bulunamadı.")
            continue
            
        print(f"\n[{subdir}] klasöründeki {len(image_files)} görüntü onarılıyor...")
        
        # Resimleri sırayla oku, onar ve kaydet
        for count, filename in enumerate(image_files, 1):
            img_path = os.path.join(input_dir, filename)
            
            try:
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"  -> UYARI: Görüntü okunamadı: {img_path}")
                    continue
                    
                # Görüntü onarma (pre-processing) adımlarını uygula
                enhanced_img = enhance_image(img)
                
                # Sonucu kaydet
                out_path = os.path.join(output_dir, filename)
                cv2.imwrite(out_path, enhanced_img)
                
            except Exception as e:
                print(f"  -> HATA: {filename} işlenirken bir sorun oluştu: {e}")
                
            # İlerleme durumu raporu
            if count % 10 == 0 or count == len(image_files):
                print(f"  - İlerleme: {count}/{len(image_files)} onarıldı.")
                
    print("\nİşlem başarıyla tamamlandı!")
    print(f"Tüm onarılmış görüntüler '{output_base_dir}' dizinine kaydedildi.")

if __name__ == "__main__":
    main()
