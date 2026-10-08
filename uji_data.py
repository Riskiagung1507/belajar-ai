import pandas as pd
import numpy as np

# Membuat data tabel sederhana pakai Pandas
data = {
    'Nama_AI': ['ChatGPT', 'Claude', 'Gemini'],
    'Tahun_Rilis': [2022, 2023, 2023]
}
tabel = pd.DataFrame(data)

print("Data Model AI:")
print(tabel)