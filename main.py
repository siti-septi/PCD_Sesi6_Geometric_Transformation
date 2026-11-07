
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# ==========================================
# 1. MEMBACA GAMBAR ASLI
# ==========================================

img = imageio.imread("gambar.jpg")

# Mengubah gambar grayscale menjadi RGB
if img.ndim == 2:
    img = np.stack((img, img, img), axis=-1)

# Menghapus channel alpha jika gambar memiliki RGBA
if img.shape[-1] == 4:
    img = img[:, :, :3]

img = img.astype(np.uint8)

print("Ukuran gambar asli:", img.shape)

# ==========================================
# 2. TRANSLATION (PERGESERAN)
# ==========================================

translation = ndimage.shift(
    img,
    shift=(50, 80, 0),
    mode="constant",
    cval=0
)

# ==========================================
# 3. SCALING (PERUBAHAN UKURAN)
# ==========================================

scaling_05 = ndimage.zoom(
    img,
    zoom=(0.5, 0.5, 1),
    order=1
)

scaling_2 = ndimage.zoom(
    img,
    zoom=(2, 2, 1),
    order=1
)

# ==========================================
# 4. ROTATION (ROTASI)
# ==========================================

rotation_45 = ndimage.rotate(
    img, 45, reshape=True, order=1, mode="constant", cval=0
)

rotation_90 = ndimage.rotate(
    img, 90, reshape=True, order=1, mode="constant", cval=0
)

rotation_180 = ndimage.rotate(
    img, 180, reshape=True, order=1, mode="constant", cval=0
)

# ==========================================
# 5. FLIP (PENCERMINAN)
# ==========================================

horizontal_flip = np.fliplr(img)

vertical_flip = np.flipud(img)

# ==========================================
# 6. MENYIMPAN HASIL TRANSFORMASI
# ==========================================

imageio.imwrite("hasil_translation.png", translation.astype(np.uint8))
imageio.imwrite("hasil_scaling_05.png", scaling_05.astype(np.uint8))
imageio.imwrite("hasil_scaling_2.png", scaling_2.astype(np.uint8))
imageio.imwrite("hasil_rotation_45.png", rotation_45.astype(np.uint8))
imageio.imwrite("hasil_rotation_90.png", rotation_90.astype(np.uint8))
imageio.imwrite("hasil_rotation_180.png", rotation_180.astype(np.uint8))
imageio.imwrite("hasil_horizontal_flip.png", horizontal_flip.astype(np.uint8))
imageio.imwrite("hasil_vertical_flip.png", vertical_flip.astype(np.uint8))

# ==========================================
# 7. MENAMPILKAN SEMUA HASIL TRANSFORMASI
# ==========================================

fig, axes = plt.subplots(3, 3, figsize=(15, 13))

hasil = [
    (img, "Gambar Asli"),
    (translation, "Translation"),
    (scaling_05, "Scaling 0.5"),
    (scaling_2, "Scaling 2"),
    (rotation_45, "Rotation 45 Derajat"),
    (rotation_90, "Rotation 90 Derajat"),
    (rotation_180, "Rotation 180 Derajat"),
    (horizontal_flip, "Horizontal Flip"),
    (vertical_flip, "Vertical Flip")
]

for ax, (gambar, judul) in zip(axes.flat, hasil):
    ax.imshow(gambar)
    ax.set_title(judul)
    ax.axis("off")

plt.suptitle(
    "HASIL GEOMETRIC TRANSFORMATION",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Menyimpan gambar gabungan
plt.savefig("hasil_transformasi_lengkap.png", dpi=300)

plt.show()

print("Semua proses transformasi geometris berhasil dilakukan!")
