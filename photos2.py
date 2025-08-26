import os
from PIL import Image

# Ścieżki
input_root = r"C:\Users\jakub\OneDrive\Pulpit\merged_images"
output_root = r"C:\Users\jakub\OneDrive\Pulpit\final_images"
os.makedirs(output_root, exist_ok=True)

# Pobierz unikalne dataset names
all_files = os.listdir(input_root)
datasets = sorted(set(f.split("_")[0] for f in all_files if f.endswith(".png")))

# Klasyfikatory w ustalonej kolejności (jeśli potrzebujesz, ale nie będziemy ich teraz używać)
classifiers = ["LR", "RF", "KNN", "SVM", "DNN"]

for dataset in datasets:
    rows = []
    max_row_width = 0
    total_height = 0

    for clf in classifiers:
        file_name = f"{dataset}_{clf}.png"
        file_path = os.path.join(input_root, file_name)

        if not os.path.exists(file_path):
            print(f"⚠️ Brak pliku: {file_name}")
            continue

        img = Image.open(file_path)
        row_height = img.height
        row_width = img.width  # Tylko szerokość wykresu

        # Dodajemy tylko wykres, bez etykiety
        rows.append(img)
        max_row_width = max(max_row_width, row_width)
        total_height += row_height

    # Tworzymy końcowy obrazek dla datasetu
    final_img = Image.new("RGB", (max_row_width, total_height), (255, 255, 255))
    y_offset = 0
    for row_img in rows:
        final_img.paste(row_img, (0, y_offset))
        y_offset += row_img.height

    # Zapisz obrazek
    out_path = os.path.join(output_root, f"{dataset}_summary.png")
    final_img.save(out_path)
    print(f"✅ Zapisano: {out_path}")
