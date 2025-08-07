import os
from PIL import Image
import numpy as np

# diretórios – ajuste conforme sua organização
input_dir = "."
output_dir = "images_swapped"
os.makedirs(output_dir, exist_ok=True)

# Sequência de trocas (src_color → dst_color), nesta ordem:
swap_sequence = [
    # 1) Amarelo → Azul
    ((128, 128,   0), (  0,   0, 255)),
    # 2) Verde   → Amarelo
    ((  0, 128,   0), (128, 128,   0)),
    # 3) Azul    → Verde
    ((  0,   0, 255), (  0, 128,   0)),
]

def swap_colors(img: Image.Image) -> Image.Image:
    out = np.array(img)  # começa com a imagem RGB original

    for (sr, sg, sb), (dr, dg, db) in swap_sequence:
        # máscara sobre o OUT, não sobre o ARR original
        mask = (
            (out[:, :, 0] == sr) &
            (out[:, :, 1] == sg) &
            (out[:, :, 2] == sb)
        )
        out[mask] = (dr, dg, db)

    return Image.fromarray(out.astype(np.uint8))

# Processa todos os PNG/JPG no diretório atual
for fname in os.listdir(input_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
    swapped = swap_colors(img)
    swapped.save(os.path.join(output_dir, fname))
    print(f"→ processado: {fname}")

print("Concluído! As imagens estão em ./swapped")