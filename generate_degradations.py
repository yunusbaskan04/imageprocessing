import cv2
import numpy as np
import os

def apply_low_light(img, gamma=2.5):
    """Görüntüyü karartmak için gamma correction uygular."""
    # Lookup table (LUT) oluşturarak işlemi hızlandırıyoruz
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_fog(img, alpha=0.5):
    """Düz gri bir katman ile sentetik sis uygular (Hızlı yöntem)."""
    # Görüntüyle aynı boyutta açık gri bir katman oluştur (RGB: 200, 200, 200)
    fog_layer = np.full(img.shape, 200, dtype=np.uint8)
    # Görüntü ve sis katmanını alpha değerine göre birleştir
    return cv2.addWeighted(img, 1 - alpha, fog_layer, alpha, 0)

def apply_gaussian_noise(img, mean=0, std=60):
    """Görüntüye Gaussian noise (rastgele gürültü) ekler."""
    # Numpy ile her kanala ayrı ayrı gürültü üretmek daha belirgin ve doğru bir etki yaratır
    noise = np.random.normal(mean, std, img.shape)
    # Görüntü ile gürültüyü topla ve 0-255 aralığına kırp (clip)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img

def main():
    # 1. Girdi klasörünü belirle
    # Kendi klasör yapınıza göre burayı 'dataset_original/val/images' olarak da değiştirebilirsiniz.
    input_dir = os.path.join("dataset_original", "test", "images")
    
    # 2. Çıktı klasörlerini belirle
    output_base_dir = "dataset_degraded"
    low_light_dir = os.path.join(output_base_dir, "low_light")
    fog_dir = os.path.join(output_base_dir, "fog")
    noise_dir = os.path.join(output_base_dir, "noise")
    
    # 3. Klasörleri güvenli bir şekilde oluştur (exist_ok=True ile varsa hata vermez)
    try:
        os.makedirs(low_light_dir, exist_ok=True)
        os.makedirs(fog_dir, exist_ok=True)
        os.makedirs(noise_dir, exist_ok=True)
        print(f"Çıktı klasörleri '{output_base_dir}' altında hazırlandı.")
    except Exception as e:
        print(f"Klasörler oluşturulurken hata meydana geldi: {e}")
        return
    
    # Girdi klasörünün var olup olmadığını kontrol et
    if not os.path.exists(input_dir):
        print(f"HATA: Girdi klasörü bulunamadı: {input_dir}")
        print("Lütfen kod içerisindeki 'input_dir' değişkenini datasetinizin bulunduğu doğru yola göre ayarlayın.")
        return

    # Okunacak desteklenen görüntü formatları
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Klasördeki tüm resim dosyalarını listele
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"UYARI: '{input_dir}' klasöründe hiç resim bulunamadı.")
        return
        
    print(f"Toplam {len(image_files)} görüntü işlenecek. Lütfen bekleyin...")
    
    # 4. Görüntüleri tek tek oku ve işlemleri uygula
    for count, filename in enumerate(image_files, 1):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"UYARI: Görüntü okunamadı, atlanıyor: {img_path}")
            continue
            
        # --- Bozulmaları Uygula ve Kaydet ---
        
        # 1. Düşük Işık
        img_low_light = apply_low_light(img, gamma=2.5)
        cv2.imwrite(os.path.join(low_light_dir, filename), img_low_light)
        
        # 2. Sentetik Sis
        img_fog = apply_fog(img, alpha=0.5)
        cv2.imwrite(os.path.join(fog_dir, filename), img_fog)
        
        # 3. Gaussian Noise
        img_noise = apply_gaussian_noise(img, mean=0, std=60)
        cv2.imwrite(os.path.join(noise_dir, filename), img_noise)
        
        # Kullanıcıya süreç hakkında bilgi ver (Her 10 resimde bir)
        if count % 10 == 0 or count == len(image_files):
            print(f"İlerleme: {count}/{len(image_files)} tamamlandı.")
            
    print("İşlem başarıyla tamamlandı!")
    print(f"Oluşturulan görüntüler şu dizinlerde bulunabilir:\n- {low_light_dir}\n- {fog_dir}\n- {noise_dir}")

if __name__ == "__main__":
    main()
