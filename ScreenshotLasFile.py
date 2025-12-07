import laspy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
def las_to_topdown_png(las_file, output_png="output.png", dpi=300, point_size=1):
    """
    Tworzy rzut z góry (XY) pliku LAS i zapisuje jako PNG.

    Args:
        las_file (str): ścieżka do pliku LAS
        output_png (str): nazwa pliku PNG do zapisu
        dpi (int): rozdzielczość obrazka
        point_size (float): wielkość punktów na obrazie
    """
    print("Wczytywanie pliku LAS...")
    las = laspy.read(las_file)

    X = las.x
    Y = las.y

    # Wybór koloru – preferowane RGB jeśli istnieją
    has_rgb = all(hasattr(las, attr) for attr in ["red", "green", "blue"])

    if has_rgb:
        print("Używam kolorów RGB...")
        R = las.red / np.max(las.red)
        G = las.green / np.max(las.green)
        B = las.blue / np.max(las.blue)
        colors = np.vstack([R, G, B]).T
    else:
        print("Brak RGB – używam intensywności...")
        intensity = las.intensity.astype(float)
        if np.max(intensity) > 0:
            intensity /= np.max(intensity)
        colors = np.vstack([intensity, intensity, intensity]).T

    # Tworzenie wykresu
    print("Generowanie obrazu PNG...")
    plt.figure(figsize=(10, 10), dpi=dpi)
    plt.scatter(X, Y, s=point_size, c=colors, marker=".", linewidths=0)

    plt.axis("equal")
    plt.axis("off")  # wyłączenie osi, żeby zdjęcie było czyste

    # Zapis PNG
    os.makedirs(os.path.dirname(output_png), exist_ok=True) if "/" in output_png else None
    plt.savefig(output_png, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Zapisano obraz do pliku: {output_png}")
    return output_png
def las_to_topdown_png(las_file, output_png="output.png", dpi=300, point_size=1):
    """
    Tworzy rzut z góry (XY) pliku LAS i zapisuje jako PNG.

    Args:
        las_file (str): ścieżka do pliku LAS
        output_png (str): nazwa pliku PNG do zapisu
        dpi (int): rozdzielczość obrazka
        point_size (float): wielkość punktów na obrazie
    """
    print("Wczytywanie pliku LAS...")
    las = laspy.read(las_file)

    X = las.x
    Y = las.y

    # Wybór koloru – preferowane RGB jeśli istnieją
    has_rgb = all(hasattr(las, attr) for attr in ["red", "green", "blue"])

    if has_rgb:
        print("Używam kolorów RGB...")
        R = las.red / np.max(las.red)
        G = las.green / np.max(las.green)
        B = las.blue / np.max(las.blue)
        colors = np.vstack([R, G, B]).T
    else:
        print("Brak RGB – używam intensywności...")
        intensity = las.intensity.astype(float)
        if np.max(intensity) > 0:
            intensity /= np.max(intensity)
        colors = np.vstack([intensity, intensity, intensity]).T

    # Tworzenie wykresu
    print("Generowanie obrazu PNG...")
    plt.figure(figsize=(10, 10), dpi=dpi)
    plt.scatter(X, Y, s=point_size, c=colors, marker=".", linewidths=0)

    plt.axis("equal")
    plt.axis("off")  # wyłączenie osi, żeby zdjęcie było czyste

    # Zapis PNG
    os.makedirs(os.path.dirname(output_png), exist_ok=True) if "/" in output_png else None
    plt.savefig(output_png, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Zapisano obraz do pliku: {output_png}")
    return output_png

def main():
    if not sys.argv[1]:
        print("Nie podano pliku .las")
        return
    las_file = sys.argv[1]
    las_to_topdown_png(las_file)
    las_to_topdown_png

if __name__ == "__main__":
    main() 