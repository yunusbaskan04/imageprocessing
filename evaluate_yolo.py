import os
import shutil
import yaml
import cv2
from ultralytics import YOLO

def setup_eval_env(images_src, labels_src, target_class_id=4):
    """
    YOLOv8'in sorunsuz çalışabilmesi için geçici bir değerlendirme klasörü oluşturur.
    Resimleri ve etiketleri buraya kopyalar.
    NOT: Projenizdeki orijinal etiketler büyük ihtimalle '0' (drone) sınıfında.
    Ancak biz hazır COCO modeli (yolov8n.pt) kullanacağımız için, COCO'daki 
    'aeroplane' (uçak) sınıfı olan '4' ID'sine etiketleri dönüştürüyoruz.
    Böylece YOLO modeli drone'ları uçak olarak tanıdığında doğru (True Positive) sayılacak.
    """
    eval_dir = "temp_eval"
    images_dir = os.path.join(eval_dir, "images")
    labels_dir = os.path.join(eval_dir, "labels")
    
    # Geçici klasörleri temizle ve yeniden oluştur
    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
    os.makedirs(images_dir)
    os.makedirs(labels_dir)
    
    # Resimleri kopyala
    for f in os.listdir(images_src):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            shutil.copy2(os.path.join(images_src, f), os.path.join(images_dir, f))
            
    # Etiketleri kopyala ve sınıf ID'sini 4'e (aeroplane) çevir
    if os.path.exists(labels_src):
        for f in os.listdir(labels_src):
            if f.endswith('.txt'):
                src_path = os.path.join(labels_src, f)
                dst_path = os.path.join(labels_dir, f)
                with open(src_path, 'r') as file_in, open(dst_path, 'w') as file_out:
                    for line in file_in:
                        parts = line.strip().split()
                        if len(parts) > 0:
                            parts[0] = str(target_class_id) # Sınıfı 4 yap
                            file_out.write(" ".join(parts) + "\n")
                            
    # data.yaml dosyasını oluştur
    yaml_path = os.path.join(eval_dir, "data.yaml")
    
    # 80 Sınıflı bir yapı oluşturup sadece 4'ü isimlendiriyoruz
    names_dict = {i: f'class_{i}' for i in range(80)}
    names_dict[target_class_id] = 'aeroplane'
    
    data = {
        'path': os.path.abspath(eval_dir),
        'train': 'images', # YOLO val modu için de train klasörünü istiyor
        'val': 'images',
        'nc': 80,
        'names': names_dict
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
        
    return yaml_path

def evaluate_and_visualize(model, name, images_src, labels_src, results_dir):
    print(f"\n{'-'*50}\n[{name}] veri seti değerlendiriliyor...\n{'-'*50}")
    
    # 1. Veri setini ve data.yaml'ı geçici dizinde ayarla
    yaml_path = setup_eval_env(images_src, labels_src, target_class_id=4)
    
    # 2. Modeli Test Et
    # Sadece aeroplane (4) sınıfını baz alıyoruz, görseller (plots) ve detaylı konsol çıktısını kapatıyoruz.
    metrics = model.val(data=yaml_path, classes=[4], verbose=False, plots=False)
    
    # Sonuçları (mAP50, Precision, Recall) çek
    mAP50 = metrics.box.map50
    p = metrics.box.mp  # Mean precision for the evaluated classes
    r = metrics.box.mr  # Mean recall for the evaluated classes
    
    # 3. 'Visual Results' için 2 tane örnek görüntüyü Bounding Box'lı olarak kaydet
    os.makedirs(results_dir, exist_ok=True)
    images = [f for f in os.listdir(images_src) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    sample_images = images[:2] # 2 örnek resim seç
    
    for img_name in sample_images:
        img_path = os.path.join(images_src, img_name)
        # Bounding box bul (sadece uçak = 4 sınıfı için, güven skoru > 0.25)
        res = model.predict(img_path, conf=0.25, classes=[4], verbose=False)
        
        # Sonucu çiz ve kaydet
        plotted_img = res[0].plot()
        out_name = f"{name}_{img_name}"
        cv2.imwrite(os.path.join(results_dir, out_name), plotted_img)
        
    return p, r, mAP50

def main():
    print("Pre-trained YOLOv8n modeli yükleniyor...")
    # Not: pip install ultralytics yüklü değilse burada hata verecektir.
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu (Muhtemelen ultralytics kütüphanesi eksik): {e}")
        return
        
    original_images = os.path.join("dataset_original", "test", "images")
    original_labels = os.path.join("dataset_original", "test", "labels")
    
    degraded_base = "dataset_degraded"
    enhanced_base = "dataset_enhanced"
    subdirs = ["low_light", "fog", "noise"]
    
    results_dir = "dataset_results"
    
    results_table = {}
    
    # === TESTLER ===
    
    # 1. Orijinal Veri Seti
    if os.path.exists(original_images):
        p, r, mAP50 = evaluate_and_visualize(model, "Original", original_images, original_labels, results_dir)
        results_table["Original"] = (p, r, mAP50)
    
    # 2. Bozulmuş (Degraded) ve İyileştirilmiş (Enhanced) Veri Setleri
    for sub in subdirs:
        # Degraded
        deg_img_dir = os.path.join(degraded_base, sub)
        deg_name = f"{sub.capitalize()}_Degraded"
        if os.path.exists(deg_img_dir):
            p, r, mAP50 = evaluate_and_visualize(model, deg_name, deg_img_dir, original_labels, results_dir)
            results_table[deg_name] = (p, r, mAP50)
            
        # Enhanced
        enh_img_dir = os.path.join(enhanced_base, sub)
        enh_name = f"{sub.capitalize()}_Enhanced"
        if os.path.exists(enh_img_dir):
            p, r, mAP50 = evaluate_and_visualize(model, enh_name, enh_img_dir, original_labels, results_dir)
            results_table[enh_name] = (p, r, mAP50)
            
    # Temizlik (Geçici klasörü sil)
    if os.path.exists("temp_eval"):
        shutil.rmtree("temp_eval")
        
    # === TABLOYU YAZDIRMA ===
    print("\n\n" + "="*75)
    print(" " * 15 + "YOLOv8n NESNE TESPİT (OBJECT DETECTION) METRİKLERİ")
    print("="*75)
    print(f"| {'Veri Seti (Dataset Variation)':<30} | {'mAP@50':<10} | {'Precision':<10} | {'Recall':<10} |")
    print("-" * 75)
    
    # Belirli bir mantıksal sırada yazdırmak için liste
    order = ["Original", "Fog_Degraded", "Fog_Enhanced", "Low_light_Degraded", "Low_light_Enhanced", "Noise_Degraded", "Noise_Enhanced"]
    
    for key in order:
        if key in results_table:
            p, r, mAP50 = results_table[key]
            print(f"| {key:<30} | {mAP50:.4f}     | {p:.4f}     | {r:.4f}     |")
            
    print("="*75)
    print(f"\n✅ Makalede kullanmak için örnek Bounding Box resimleri '{results_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()
