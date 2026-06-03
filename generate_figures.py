import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os

def draw_pipeline():
    """Boru hattının blok diyagramını çizer."""
    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.axis('off')
    
    steps = [
        "Raw Image\n(Degraded)",
        "RGB to LAB\nColor Space",
        "CLAHE\n(L-Channel Only)",
        "LAB to BGR\nColor Space",
        "Bilateral Filter\n(Edge Preserving)",
        "Enhanced\nImage\n(Output)"
    ]
    
    box_width = 1.6
    box_height = 0.9
    spacing = 0.5
    start_x = 0.2
    y = 0.5
    
    for i, step in enumerate(steps):
        x = start_x + i * (box_width + spacing)
        
        # Kutuyu çiz
        rect = patches.FancyBboxPatch(
            (x, y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.1", 
            linewidth=2, edgecolor='#2c3e50', facecolor='#ecf0f1'
        )
        ax.add_patch(rect)
        
        # Metni ekle
        ax.text(x + box_width/2, y, step, ha='center', va='center', 
                fontsize=11, fontweight='bold', family='sans-serif', color='#2c3e50')
        
        # Oku çiz (son kutu hariç)
        if i < len(steps) - 1:
            arrow_x = x + box_width
            ax.annotate('', xy=(arrow_x + spacing, y), xytext=(arrow_x, y),
                        arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.5, color='#e74c3c'))
                        
    ax.set_xlim(0, start_x + len(steps) * (box_width + spacing))
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('fig1_pipeline.png', dpi=300, bbox_inches='tight')
    print("fig1_pipeline.png (Akis Semasi) basariyla olusturuldu.")

def create_subplot(img_paths, titles, out_filename):
    """3'lü yan yana subplot oluşturur."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (path, title) in enumerate(zip(img_paths, titles)):
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
            else:
                axes[i].text(0.5, 0.5, 'Image Load Error', ha='center', va='center')
        else:
            axes[i].text(0.5, 0.5, 'Image Not Found', ha='center', va='center')
            
        axes[i].set_title(title, fontsize=16, fontweight='bold', pad=10)
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    print(f"{out_filename} basariyla olusturuldu.")

def generate_visual_results():
    orig_dir = os.path.join("dataset_original", "test", "images")
    deg_dir = os.path.join("dataset_degraded", "fog")
    enh_dir = os.path.join("dataset_enhanced", "fog")
    
    sample_img = None
    if os.path.exists(deg_dir):
        images = [f for f in os.listdir(deg_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            sample_img = images[0] # İlk resmi örnek al
            
    if sample_img:
        paths = [
            os.path.join(orig_dir, sample_img),
            os.path.join(deg_dir, sample_img),
            os.path.join(enh_dir, sample_img)
        ]
        titles = ['(a) Original Image', '(b) Degraded (Synthetic Fog)', '(c) Enhanced (Proposed Pipeline)']
        create_subplot(paths, titles, 'fig2_visual_results.png')
    else:
        print("HATA: fig2_visual_results.png için örnek resim bulunamadı.")

def generate_yolo_results():
    yolo_dir = "dataset_results"
    if os.path.exists(yolo_dir):
        res_images = os.listdir(yolo_dir)
        base_name = None
        for f in res_images:
            if f.startswith("Original_"):
                base_name = f.replace("Original_", "")
                break
                
        if base_name:
            paths = [
                os.path.join(yolo_dir, f"Original_{base_name}"),
                os.path.join(yolo_dir, f"Fog_Degraded_{base_name}"),
                os.path.join(yolo_dir, f"Fog_Enhanced_{base_name}")
            ]
            titles = ['(a) YOLO Detection (Original)', '(b) YOLO Detection (Fog Degraded)', '(c) YOLO Detection (Fog Enhanced)']
            create_subplot(paths, titles, 'fig3_yolo_results.png')
        else:
            print("HATA: fig3_yolo_results.png için Original prefix'li resim bulunamadı.")
    else:
        print("HATA: dataset_results klasörü bulunamadı.")

if __name__ == "__main__":
    print("Makale görselleri (Fig 1, 2, 3) hazırlanıyor...")
    draw_pipeline()
    generate_visual_results()
    generate_yolo_results()
    print("Tüm görseller hazır!")
