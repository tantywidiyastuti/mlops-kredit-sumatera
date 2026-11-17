# CATATAN EKSPLORASI YANG BERORIENTASI KEPUTUSAN
**Mata Kuliah:** Machine Learning Operations (SD25-40001)  
**Praktikum 1:** Menyiapkan Repositori dan Mengenal Data  
**Dataset:** Kredit Sumatera (`--seed 42`, 32.400 baris, Periode 2022-01 s.d. 2024-12)  

---

## 1. Empat Pertanyaan Eksplorasi Berdasarkan Data

Eksplorasi data pada tahap ini tidak ditujukan untuk sekadar memvisualisasikan data, melainkan untuk menjawab empat pertanyaan mendasar yang secara langsung menentukan keputusan teknis dan operasional pada tahapan MLOps berikutnya.

### (1) Seberapa tidak seimbang targetnya?
* **Bukti Data:** Dari total **32.400** permohonan pinjaman pada portofolio, terdapat **2.690** debitur berstatus macet (`gagal_bayar = 1`) dan **29.710** debitur berstatus lancar (`gagal_bayar = 0`). Tingkat gagal bayar portofolio tercatat tepat **8,30%** (rasio kelas $\approx 1:11$).
* **Keputusan Teknis & Bisnis:**
  1. *Metrik Evaluasi*: Metrik akurasi standar (*accuracy*) dilarang digunakan karena memberikan sinyal palsu; model trivial yang memprediksi seluruh nasabah berstatus lancar akan memperoleh akurasi 91,70% meski gagal mendeteksi seluruh risiko. Evaluasi performa model wajib menggunakan **ROC-AUC** (daya pembeda peringkat), **KS-Statistik**, dan koefisien **Gini**.
  2. *Ambang Batas Peka-Biaya (Cost-Sensitive Thresholding)*: Ambang batas klasifikasi tidak boleh ditetapkan pada default $0{,}5$. Di dunia perbankan, kerugian akibat meloloskan nasabah macet (*False Negative*) mencapai **Rp 12.000.000** (kehilangan pokok pinjaman), sedangkan biaya menolak nasabah baik (*False Positive*) hanya **Rp 1.500.000** (kehilangan margin bunga). Ambang batas optimal diturunkan dari kurva biaya bisnis, bergeser ke kisaran **0,0850**, sehingga model lebih agresif dalam menandai potensi gagal bayar.

### (2) Apakah populasi berubah sepanjang waktu?
* **Bukti Data:** Terjadi pergeseran komposisi populasi (*covariate shift*) yang signifikan antara periode awal (2022) hingga periode akhir (2024), dengan akselerasi pergeseran dimulai pada bulan ke-22 ($t \ge 60\%$ rentang waktu data):
  * **Pegawai Tetap** (*low risk*, laju gagal bayar historis 5,75%): Turun drastis dari **42,1%** (2022) dan 41,9% (2023) menjadi **33,9%** (2024). Pada 6 bulan terakhir, porsinya tersisa hanya 31,1%.
  * **Segmen Berisiko Tinggi** (*high risk*, laju gagal bayar $>10\%$):
    * *Nelayan* (laju gagal bayar 11,27%): Melonjak dari **6,7%** (2022/2023) menjadi **9,9%** (2024), mencapai 11,1% di akhir.
    * *Petani* (laju gagal bayar 11,00%): Meningkat dari **13,0%** (2022/2023) menjadi **16,0%** (2024), mencapai 16,7% di akhir.
    * *Pedagang Musiman* (laju gagal bayar 10,24%): Naik dari **13,9%** (2022) menjadi **17,1%** (2024).
  * Secara gabungan, porsi segmen berisiko tinggi melesat dari **33,6%** (2022) menjadi **43,0%** (2024).
  * Pergeseran demografis ini diakselerasi oleh penetrasi kanal: transaksi melalui kanal **daring** melonjak lebih dari 3 kali lipat dari **12,24%** (2022) menjadi **37,98%** (2024), menggusur dominasi kanal kantor cabang (turun dari 61,93% ke 38,06%), diikuti penurunan usia rerata debitur dari 39,0 tahun ke 35,8 tahun.
* **Keputusan Teknis & Bisnis:** Model yang dilatih pada kondisi pasar lama (2022–2023) akan mengalami degradasi kinerja (*model aging/decay*) saat diterjunkan pada populasi 2024. Hal ini mewajibkan evaluasi model menggunakan partisi temporal *Out-of-Time* (OOT) serta pemasangan sistem pemantauan pergeseran data (*drift detection*) berkala.

### (3) Kolom mana yang punya nilai hilang, dan mengapa?
* **Bukti Data:** Terdapat tiga kolom yang memiliki nilai hilang dengan pola yang sangat kontras:
  * `status_restrukturisasi`: **94,79%** kosong (30.712 baris) — merupakan fitur bocor pasca-kejadian.
  * `lama_usaha_tahun`: **2,94%** kosong (954 baris), tersebar acak merata di seluruh segmen ($2{,}5\% - 3{,}0\%$) yang tergolong *Missing Completely At Random* (MCAR).
  * `rata_saldo_90h`: **2,31%** kosong secara total (750 baris), namun terdistribusi secara *Missing Not At Random* (MNAR) / *informative missingness*:
    * Segmen **Nelayan**: **10,80%** nilai hilang (273 dari 2.528 nasabah).
    * Segmen **Petani**: **10,51%** nilai hilang (477 dari 4.538 nasabah).
    * Segmen **Pegawai Tetap**, **Wiraswasta**, dan **Pedagang Musiman**: **0,00%** nilai hilang (0 kasus).
* **Keputusan Teknis & Bisnis:** 
  * *Mengapa*: Karakteristik sosio-ekonomi masyarakat agraris dan pesisir di Sumatera beroperasi dalam ekonomi berbasis kas (*cash-driven economy*) yang minim interaksi perbankan formal harian (*unbanked/underbanked*).
  * *Tindakan*: Mengisi nilai hilang dengan median global populasi adalah bias fatal karena memperlakukan petani/nelayan seolah memiliki profil perbankan pegawai kantor. Mengisi angka 0 juga salah karena saldo Rp 0 adalah kondisi saldo tabungan yang valid (bukan ketiadaan data). Transformasi harus memisahkan penanganan nilai hilang berbasis artefak per segmen yang dipelajari dari data latih (`src/features.py`), serta dipagari oleh uji gerbang kontrak data (`MAKS_HILANG = {"rata_saldo_90h": 0.20}`) di pipeline CI.

### (4) Adakah kolom yang mencurigakan?
* **Bukti Data:** Kolom `status_restrukturisasi` terindikasi secara mutlak sebagai **kebocoran target (target leakage)**:
  * Pada debitur lancar (`gagal_bayar = 0`): **0% terisi** (seluruh 29.710 baris bernilai kosong/NaN).
  * Pada debitur macet (`gagal_bayar = 1`): **62,75% terisi** teks `"ya"` (1.688 dari 2.690 baris).
  * Dampak pada model: Model garis dasar bersih menghasilkan AUC out-of-time sebesar **0,7744**. Bila kolom ini disertakan, AUC melonjak artifisial menjadi **0,9063** ($>0{,}90$).
* **Keputusan Teknis & Bisnis:** Restrukturisasi pinjaman adalah proses kolektibilitas perbankan yang terjadi *setelah* debitur menunggak dan gagal bayar. Pada saat permohonan kredit baru diajukan ($t=0$), kolom ini belum pernah terisi di sistem operasional. Fitur ini wajib diblokir secara permanen dari matriks pemodelan melalui daftar hitam `TERLARANG` di kode sumber dan gerbang CI otomatis.

---

## 2. Dua Temuan yang Mengubah Keputusan Rancangan

Dua temuan utama dari eksplorasi ini mengubah fondasi perancangan arsitektur pipeline MLOps pada praktikum-praktikum berikutnya:

### Temuan 1: Nilai Hilang Terkondisi Segmen pada `rata_saldo_90h` Menuntut Pemisahan Transformasi Stateless vs Stateful dan Feature Store Terstandarisasi
* **Alasan dan Fakta Angka:** Tingkat *missing value* terkonsentrasi 100% pada nasabah nelayan (10,80%) dan petani (10,51%). Penggunaan fungsi imputasi sepihak di notebook (seperti `fillna(0)` atau `SimpleImputer`) menyebabkan *training-serving skew*. Perbedaan perlakuan kecil (pelatihan menggunakan median data latih, sedangkan tim deployment API mengisi nol) tidak memicu pesan galat pemrograman, namun langsung mengubah keputusan kredit pada 66 nasabah (1,22% dari total pengajuan uji), yang pada skala bank dengan 200.000 permohonan setara dengan 2.400 nasabah yang keputusannya salah secara diam-diam.
* **Perubahan Keputusan Rancangan:**
  1. *Praktikum 3 & 7 (Data Contract)*: Membuat fungsi `validasi()` di `src/data.py` dengan parameter toleransi batas hilang per kolom (`MAKS_HILANG = {"rata_saldo_90h": 0.20}`). Pipeline pelatihan akan langsung dihentikan (*fail-fast*) dengan kode galat `sys.exit(2)` apabila skema data sumber menyimpang.
  2. *Praktikum 5 (Feature Architecture)*: Menolak penulisan transformasi fitur secara ganda. Transformasi diisolasi ke dalam `src/features.py`, memisahkan transformasi *stateless* (kalkulasi deterministik baris seperti `rasio_angsuran`) dari transformasi *stateful* (artefak statistik dari data latih). Nilai median latih dibekukan ke dalam artefak `artefak_fitur.json` dan diimpor langsung oleh endpoint API layanan (`src/api.py`).

### Temuan 2: Pergeseran Populasi Dinamis (Populasi Risiko Naik dari 33,6% ke 43,0%) dan Keruntuhan Validitas Pembagian Acak
* **Alasan dan Fakta Angka:** Evaluasi menggunakan pembagian acak (*random stratified split*) menghasilkan estimasi AUC sebesar **0,7631 – 0,7900**. Namun, angka ini merupakan estimasi semu (*optimism bias*) karena pembagian acak membiarkan model mengintip pola dari periode waktu yang sama dengan data uji. Pada kenyataannya, portofolio mengalami *population drift* berat (porsi segmen berisiko tinggi bertambah 9,4 poin persentase dan kanal daring melonjak ke 37,98%). Ketika diuji dengan data masa depan yang jujur, performa model garis dasar berada di **0,7744**, dan bahkan segmen petani hanya mencatatkan AUC **0,7040** (paling rentan).
* **Perubahan Keputusan Rancangan:**
  1. *Praktikum 2 & 4 (Temporal Splitting)*: Menghapus seluruh skema `train_test_split` acak dan `k-fold cross validation` biasa. Seluruh perancangan pipeline wajib menggunakan pembagian *Out-of-Time* (OOT) dengan batas waktu yang dikonfigurasi pada `conf/params.yaml` (`batas_latih: "2023-12"`, `batas_validasi: "2024-06"`, dan `uji_oot: > "2024-06"`).
  2. *Praktikum 4 & 6 (Subpopulation Metric Logging)*: Mewajibkan pencatatan metrik performa terdisagregasi per subpopulasi (`metrik_segmen`) ke dalam `metadata.json` dan MLflow, guna mencegah ilusi performa tinggi di tingkat agregat yang menutupi diskriminasi kinerja pada kelompok rentan.
  3. *Praktikum 10 & 11 (Drift Monitoring & Retraining Policy)*: Menjadikan temuan pergeseran ini sebagai basis pemicu pelatihan ulang otomatis (*automated retraining trigger*) menggunakan uji statistik Kolmogorov-Smirnov / PSI ketika terjadi degradasi performa bulanan.

---

## 3. Daftar Kolom Bermasalah Beserta Mekanisme Masalahnya

| Nama Kolom | Status / Jenis Masalah | Mekanisme Masalah dan Konsekuensi Pemodelan |
| :--- | :--- | :--- |
| **`status_restrukturisasi`** | **Target Leakage** (Kebocoran Pasca-Keputusan) | Kolom ini hanya terisi `"ya"` pada debitur yang terbukti gagal bayar (62,75%) dan 100% kosong pada debitur lancar. Restrukturisasi adalah tindakan mitigasi kredit setelah debitur menunggak. Pada saat nasabah baru mengajukan pinjaman ($t=0$), informasi ini belum ada di sistem sumber. Memakainya sebagai fitur membuat model belajar jalan pintas (*shortcut learning*), mendongkrak AUC artifisial ke $>0{,}90$, namun model akan gagal total di lingkungan produksi (*model collapse*). |
| **`rata_saldo_90h`** | **Informative Missingness (MNAR)** & Rentan Kebocoran Waktu | Nilai hilang terkonsentrasi eksklusif pada segmen maritim dan agraris (nelayan 10,80%, petani 10,51%) akibat ketiadaan rekening formal (*unbanked*). Pengisian menggunakan median global atau nilai 0 akan mendistorsi profil risiko. Selain itu, pada agregasi basis data, jika penarikan saldo tidak membatasi rentang temporal secara ketat ($< \text{tanggal\_pengajuan}$), transaksi masa depan akan merembes ke data latih (*point-in-time leakage*). |
| **`jumlah_tunggakan_12b`** | **Sentinel Value (-1)** / Non-Ordinal Encoding | Kolom ini memuat nilai `-1` sebanyak 6.907 baris (21,32%). Nilai `-1` bukan berarti jumlah tunggakan di bawah nol, melainkan kode sandi nasabah tanpa riwayat biro kredit (SLIK). Algoritma numerik/linear akan salah mengartikan `-1` sebagai kondisi lebih baik dari 0 tunggakan, padahal data empiris membuktikan nasabah tanpa riwayat biro kredit memiliki risiko gagal bayar jauh lebih tinggi (**12,13%** vs **7,26%** pada nasabah berhistori). Wajib direkayasa menjadi fitur biner `tanpa_riwayat_biro` dan pembersihan batas bawah $\max(0, \text{tunggakan})$. |
| **`pengajuan_id`** | **High-Cardinality Identifier** | Merupakan kunci primer arbitrer berformat string unik (`YYYY-MM-XXXXX`) sebanyak 32.400 nilai. Jika lolos ke dalam matriks fitur, algoritma pohon keputusan atau linear dapat menghafal ID spesifik atau memanfaatkan urutan indeks, menyebabkan *overfitting* ekstrem tanpa kemampuan generalisasi. |
| **`tanggal_pengajuan` & `periode`** | **Serial Temporal Leakage** | Merupakan metadata sekuensial kronologis. Bila dimasukkan langsung sebagai prediktor, model akan mempelajari korelasi semu terhadap tanggal kalender tertentu. Kolom ini harus dibatasi fungsinya hanya sebagai partisi data temporal (*OOT split*) dan pelacak tren *drift*. |
| **`angsuran_bulanan`, `plafon_diajukan`, `tenor_bulan`, `bunga_tahunan`** | **Multikolinearitas Deterministik** & Variasi Skala | Nilai `angsuran_bulanan` diturunkan secara deterministik dari formula amortisasi bunga terhadap plafon dan tenor: $\text{plafon} \times (1 + \text{bunga} \times \text{tenor}/12) / \text{tenor}$. Memasukkan seluruh variabel pembentuk bersamaan dalam satuan moneter puluhan juta rupiah menciptakan multikolinearitas sempurna dan ketidakstabilan koefisien. Hubungan risiko riil harus diekstraksi ke bentuk rasio kemampuan bayar (*Debt Service Ratio* / `rasio_angsuran = angsuran / penghasilan`). |

---

## 4. Analisis Reproduksibilitas: Mengapa Notebook Saat Ini Belum Dapat Diulang

Meskipun skrip pada `notebooks/01_eksplorasi.ipynb` berhasil dieksekusi dan menghasilkan angka, status pekerjaan ini **belum dapat disebut dapat diulang (reproducible)** dalam kaidah rekayasa MLOps standar produksi. Terdapat **tiga pilar reproduksibilitas** yang belum terpenuhi:

```
+----------------------------------------------------------------------------------------------------+
|                                TIGA PILAR REPRODUKSIBILITAS MLOPS                                  |
+------------------------------------+--------------------------------+------------------------------+
|     1. DETERMINISME LINGKUNGAN     |     2. DETERMINISME KODE &     |    3. PENAUTAN VERSI KODE    |
|       (Environment Determinism)    |         ALUR EKSEKUSI          |           DAN DATA           |
|                                    |    (Code & Execution Order)    |    (Data-Code Co-Versioning)|
+------------------------------------+--------------------------------+------------------------------+
| [Status Saat Ini]:                 | [Status Saat Ini]:             | [Status Saat Ini]:           |
| Belum terisolasi dari OS lokal.    | Kode monolitik di notebook.    | Data CSV diabaikan Git       |
| Perbedaan platform (Windows CRLF   | Urutan eksekusi tersimpan di   | (.gitignore), namun belum    |
| vs Linux LF) & dependensi runner   | memori kernel/kepala analis.   | dilacak oleh DVC. Snapshot   |
| belum diuji dalam pipeline CI.     | Belum berwujud modul/DAG.      | data & commit kode terputus. |
+------------------------------------+--------------------------------+------------------------------+
```

### 1. Pilar Pertama: Determinisme Lingkungan Komputasi (Environment Reproducibility)
* **Kondisi Saat Ini**: Eksperimen dijalankan pada Python sistem lokal pengguna. Walaupun berkas `requirements.lock` telah dibuat, lingkungan tersebut belum sepenuhnya terisolasi dan belum diverifikasi secara independen oleh sistem otomatis.
* **Risiko Kegagalan**: Menjalankan kode pada sistem operasi berbeda (misalnya mesin CI berbasis `ubuntu-latest` vs mesin lokal Windows 11) rentan mengalami kegagalan akibat perbedaan format akhir baris (*line ending* CRLF vs LF), perbedaan pustaka C pendukung (*BLAS/LAPACK* pada NumPy/SciPy), serta perbedaan versi minor dependensi jika mahasiswa menginstal pustaka menggunakan `pip install <nama_paket>` tanpa merujuk ke lockfile.
* **Persyaratan Pemenuhan**: Mengunci seluruh versi pustaka eksak pada `requirements.lock`, menjalankan kode di dalam lingkungan virtual terisolasi (`.venv` atau Conda `mlops310`), serta memvalidasi kesiapan lingkungan melalui alur kerja otomatis GitHub Actions (Praktikum 7) dan kontainerisasi Docker (Praktikum 8).

### 2. Pilar Kedua: Determinisme Kode dan Alur Eksekusi (Code & Execution Determinism)
* **Kondisi Saat Ini**: Logika pemrosesan data, rekayasa fitur, pemisahan sampel, dan pelatihan model masih bercampur aduk di dalam sel-sel notebook Jupyter interaktif (`.ipynb`).
* **Risiko Kegagalan**: Jupyter Notebook bersifat *stateful*; variabel global disimpan di dalam memori kernel dan urutan eksekusi sangat bergantung pada interaksi manusia (sel mana yang dijalankan terlebih dahulu, apakah ada sel yang dijalankan ulang atau dilewati). Kode notebook tidak memiliki deklarasi kebergantungan (*dependencies*) antar-tahap yang eksplisit. Jika orang lain membuka notebook dan mengeksekusi *Run All*, urutan eksekusi belum tentu sama dengan alur berpikir penulis aslinya. Selain itu, parameter pemodelan dan batas waktu masih ditanam langsung (*hardcoded*) di dalam kode.
* **Persyaratan Pemenuhan**: Memindahkan seluruh logika dari notebook ke modul-modul Python terpisah yang dapat diimpor dan diuji (`src/data.py`, `src/features.py`, `src/train.py` pada Praktikum 3 & 4), mengekstraksi parameter ke berkas konfigurasi `conf/params.yaml`, serta mengikat rangkaian kerja ke dalam bentuk *Directed Acyclic Graph* (DAG) yang deterministik melalui `Makefile` dan `dvc.yaml`.

### 3. Pilar Ketiga: Penautan Versi Kode dan Data (Data-Code Co-Versioning & Immutability)
* **Kondisi Saat Ini**: Mengikuti praktik baik MLOps awal, berkas data berukuran besar `data/raw/*.csv` telah dikecualikan dari Git melalui `.gitignore` untuk mencegah pembengkakan riwayat repositori. Namun, pada praktikum pertama ini, perkakas pelacak data seperti **DVC (Data Version Control)** belum diinisialisasi.
* **Risiko Kegagalan**: Hubungan antara kode pemodelan dan data pelatihan terputus (*broken lineage*). Perintah `git commit` hanya merekam perubahan kode, tetapi tidak merekam sidik jari (*hash*) data yang digunakan. Jika berkas `kredit_sumatera.csv` diubah, ditimpa dengan generator berukuran 43.200 baris, atau barisnya bergeser, menjalankan kembali kode notebook pada commit yang sama akan menghasilkan nilai AUC yang berbeda drastis tanpa kemampuan audit. Git sendirian tidak dapat memulihkan keadaan data masa lalu.
* **Persyaratan Pemenuhan**: Menginisialisasi DVC (`dvc init`), melacak data menggunakan pointer hash MD5 (`kredit_sumatera.csv.dvc`), menautkan commit pointer data ke commit Git (`git tag v0.1-data-awal`), dan memastikan bahwa kombinasi perintah `git checkout <tag>` diikuti `dvc checkout` mampu memulihkan kode dan data secara simultan ke satu titik waktu yang identik (Praktikum 3 Langkah 5).
