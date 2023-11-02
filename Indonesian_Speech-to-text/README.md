Ini merupakan folder yang mencakup keseluruhan proses preprocess dan training. Sebelum memulai, pastikan untuk memastikan bahwa library ini telah terinstall:
1. cudatoolkit=11.8
2. CUDA v11.8
3. CUDNN for CUDAv11.8
4. PyTorch
5. Torchaudio
6. Torchvision
7. Pandas
8. Numpy
9. Matplotlib
10. pydub

Kemudian jalankan preprocess_data.py dalam folder dataset untuk mendownload dataset (semua lisensi dan hak cipta dimiliki oleh Mozilla Common Voice). Pastikan untuk menginstall `gdown` terlebih dahulu.
```bash
pip install gdown
cd dataset
python prepare_dataset.py
```

**Author:**
* Dr. Rizka Wakhidatus Sholikah, S.Kom.* (Departemen Teknologi Informasi ITS)
* Kevin Putra Santoso (Departemen Teknologi Informasi ITS)
* Mohammad Idris Arif Budiman (Departemen Teknik Informatika ITS)
