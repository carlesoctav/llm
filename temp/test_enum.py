from enum import auto, StrEnum


class Warna(StrEnum):
    MERAH = auto()  # Nilainya otomatis menjadi "merah" (lowercase dari nama variabel)
    BIRU = "biru"  # Nilai ditentukan secara manual


# Perbandingan langsung (Ini akan menghasilkan True)
print("merah" == Warna.MERAH)
print("biru" == Warna.BIRU)
