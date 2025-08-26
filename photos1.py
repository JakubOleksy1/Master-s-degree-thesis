import os
from PIL import Image, ImageDraw, ImageFont

input_root = r"C:\Users\jakub\OneDrive\Pulpit\images"
output_root = r"C:\Users\jakub\OneDrive\Pulpit\merged_images"
os.makedirs(output_root, exist_ok=True)

# Mapa klasyfikatorów - skróty na pełne nazwy
classifier_names = {
    "LR": "Logistic Regression",
    "RF": "Random Forest",
    "KNN": "K-Nearest Neighbors",
    "SVM": "Support Vector Machine",
    "DNN": "Deep Neural Network"
}

# Ustawienia
label_width = 300

try:
    font = ImageFont.truetype("arial.ttf", 26)
except:
    font = ImageFont.load_default()

def merge_and_label_images(roc_path, cm_path, dataset, classifier, out_path):
    try:
        roc_img = Image.open(roc_path)
        cm_img = Image.open(cm_path)

        # Wyrównanie wysokości
        max_height = max(roc_img.height, cm_img.height)
        roc_img = roc_img.resize((roc_img.width, max_height))
        cm_img = cm_img.resize((cm_img.width, max_height))

        # Obraz bez górnego tytułu, tylko poziomo
        total_width = label_width + roc_img.width + cm_img.width
        merged_img = Image.new("RGB", (total_width, max_height), (255, 255, 255))

        draw = ImageDraw.Draw(merged_img)

        # Podpis klasyfikatora po lewej
        full_classifier_name = classifier_names.get(classifier, classifier)  # Zamiast "DNN" wyświetli pełną nazwę
        bbox = draw.textbbox((0, 0), full_classifier_name, font=font)
        text_height = bbox[3] - bbox[1]
        text_y = (max_height - text_height) // 2
        draw.text((10, text_y), full_classifier_name, fill="black", font=font)

        # Wklej ROC + CM
        merged_img.paste(roc_img, (label_width, 0))
        merged_img.paste(cm_img, (label_width + roc_img.width, 0))

        merged_img.save(out_path)
        print(f"✅ Zapisano: {out_path}")

    except Exception as e:
        print(f"❌ Błąd przy {dataset}/{classifier}: {e}")

# Iteracja
for dataset in os.listdir(input_root):
    dataset_path = os.path.join(input_root, dataset)
    if not os.path.isdir(dataset_path):
        continue

    for classifier in os.listdir(dataset_path):
        clf_path = os.path.join(dataset_path, classifier)
        if not os.path.isdir(clf_path):
            continue

        roc_path = os.path.join(clf_path, "roc.png")
        cm_path = os.path.join(clf_path, "confusion.png")
        out_file = os.path.join(output_root, f"{dataset}_{classifier}.png")

        if os.path.exists(roc_path) and os.path.exists(cm_path):
            merge_and_label_images(roc_path, cm_path, dataset, classifier, out_file)
        else:
            print(f"⚠️ Brak plików dla {dataset}/{classifier}")
