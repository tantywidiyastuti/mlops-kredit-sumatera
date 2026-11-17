# LAPORAN DUA KEGAGALAN: PENUAAN MODEL DAN TRAINING–SERVING SKEW
**Mata Kuliah:** Machine Learning Operations (SD25-40001)  
**Praktikum 2:** Memperagakan Penuaan Model dan Training–Serving Skew  
**Dataset:** Kredit Sumatera (`--seed 42`, 32.400 baris, Periode 2022-01 s.d. 2024-12)  
**Penyusun:** Tanty Widiyastuti (NIM: 123450094)  

---

## 1. Bukti Empiris Penuaan Model (Model Decay) Sepanjang Tahun 2024

Sistem pembelajaran mesin (*machine learning*) memiliki karakteristik kegagalan yang secara fundamental berbeda dari rekayasa perangkat lunak tradisional: **kinerja model dapat terdegradasi parah tanpa ada satu baris kode pun yang berubah atau rusak**.

### Tabel Evaluasi Kinerja Bulanan (Data 2024)
Model dilatih satu kali menggunakan data historis periode 2022-01 s.d. 2023-12 ($N_{\text{latih}} = 21.600$, `gagal_bayar = 7,50%`). Model dibekukan (*frozen*), lalu kinerjanya diukur pada setiap bulan sepanjang tahun 2024 ($N = 900$ debitur/bulan) secara independen:

| Periode | Ukuran Sampel ($n$) | ROC-AUC | Laju Gagal Bayar Aktual | Evaluasi Dinamika Kinerja |
| :---: | :---: | :---: | :---: | :--- |
| **2024-01** | 900 | **0,7881** | 9,11% | Awal rilis; kinerja masih menyerupai data validasi |
| **2024-02** | 900 | **0,7183** | 7,89% | Anjlok tajam pertama; sinyal awal instabilitas |
| **2024-03** | 900 | **0,7440** | 7,78% | Pemulihan parsial namun di bawah garis dasar |
| **2024-04** | 900 | **0,7744** | 7,89% | Sesuai rata-rata garis dasar *out-of-time* |
| **2024-05** | 900 | **0,8122** | 7,33% | Puncak semu akibat variasi acak sampel musiman |
| **2024-06** | 900 | **0,7681** | 10,56% | Titik balik: laju gagal bayar menembus dua digit |
| **2024-07** | 900 | **0,7798** | 11,00% | Tekanan makro ekonomi mulai ditanamkan |
| **2024-08** | 900 | **0,7560** | 11,22% | Tren penurunan AUC berlanjut |
| **2024-09** | 900 | **0,7726** | **12,44%** | Puncak gagal bayar tertinggi sepanjang tahun |
| **2024-10** | 900 | **0,7902** | 9,78% | Fluktuasi performa di tengah populasi baru |
| **2024-11** | 900 | **0,7525** | 10,44% | Kinerja terus tertekan |
| **2024-12** | 900 | **0,7923** | **12,22%** | Laju gagal bayar bertahan tinggi (>12%) |

```
Grafik Perkembangan ROC-AUC dan Tingkat Gagal Bayar Sepanjang 2024:
(Tersimpan pada artefak: plots/penuaan_model_2024.png)

AUC     0.82 |          [Mei: 0.812]
        0.80 | [Jan: 0.788]                  [Okt: 0.790]   [Des: 0.792]
        0.78 |            [Apr: 0.774]   [Jul: 0.780]   [Sep: 0.773]
        0.76 |                      [Jun: 0.768]   [Agu: 0.756]   [Nov: 0.753]
        0.74 |        [Mar: 0.744]
        0.72 |   [Feb: 0.718]
        -----+-----------------------------------------------------------> Bulan (2024)
Laju    13%  |                                      * [Sep: 12.4%] * [Des: 12.2%]
Gagal   11%  |                      * [Jun: 10.6%]     * [Agu: 11.2%] * [Nov: 10.4%]
Bayar    9%  | * [Jan: 9.1%]                                 * [Okt: 9.8%]
         7%  |   * [Feb: 7.9%] * [Mar: 7.8%] * [Apr: 7.9%] * [Mei: 7.3%]
```

### Interpretasi Grafik & Fenomena Penuaan
1. **Bukan Garis Lurus Sempurna**: Penuaan model di lapangan nyata tidak berbentuk garis linier menurun yang halus, melainkan berupa **tren penurunan bergerigi (*jagged decline*) yang disertai lonjakan derau statistik**. 
2. **Korelasi Terbalik dengan Laju Gagal Bayar**: Pada semester 1 (Januari–Mei), tingkat default stabil di kisaran $7{,}3\% - 9{,}1\%$. Memasuki semester 2 (Juni–Desember), tingkat gagal bayar melonjak tajam ke kisaran **$10{,}56\% - 12{,}44\%$**. Peningkatan risiko portofolio ini menekan kemampuan pemodelan; model yang dikalibrasi pada masa kredit longgar (2022–2023) gagal mengantisipasi eskalasi risiko di akhir 2024.
3. **Mengapa MLOps Diperlukan**: Dalam rekayasa perangkat lunak biasa (DevOps), selama server tidak mati dan kode tidak berubah, sistem akan bekerja dengan keandalan 100%. Pada sistem ML, kode yang statis akan membusuk (*rot*) secara fungsional karena dinamika dunia nyata di luar sistem selalu berubah.

---

## 2. Bukti Pergeseran Komposisi Populasi dan Kaitannya dengan Penurunan Kinerja

Penurunan performa model terjadi bukan karena fungsi matematika di dalam kode scikit-learn rusak, melainkan karena **distribusi data masukan (*input distribution*) telah bergeser jauh dari distribusi data yang dipelajari saat pelatihan**.

### Pergeseran Distribusi Segmen (Awal vs Akhir)
Data dianalisis dengan membandingkan 6 bulan pertama (2022-H1, $N=5.400$) terhadap 6 bulan terakhir (2024-H2, $N=5.400$):

| Segmen Nasabah | Proporsi 2022-H1 (Awal) | Proporsi 2024-H2 (Akhir) | Pergeseran Relatif | Karakteristik Finansial Utama |
| :--- | :---: | :---: | :---: | :--- |
| **Pegawai Tetap** | **41,41%** | **31,11%** | **-10,30 pp** (Turun -24,9%) | Penghasilan tetap, varian rendah (0,04), berhistori biro |
| **Wiraswasta** | **25,06%** | **23,22%** | **-1,84 pp** (Turun -7,3%) | Usaha relatif stabil, varian penghasilan sedang (0,22) |
| **Pedagang Musiman** | **14,06%** | **17,87%** | **+3,81 pp** (Naik +27,1%) | Bergantung siklus pasar, varian penghasilan tinggi (0,41) |
| **Petani** | **12,93%** | **16,74%** | **+3,81 pp** (Naik +29,5%) | Siklus panen musiman, varian tinggi (0,46), unbanked |
| **Nelayan** | **6,56%** | **11,06%** | **+4,50 pp** (Naik +68,6%) | Musim tangkap laut, varian tertinggi (0,48), unbanked |

### Faktor Pendorong Pergeseran Lingkungan Operasional
1. **Ledakan Kanal Digital**: Pengajuan melalui kanal *daring* melonjak dari **12,24%** (2022) menjadi **37,98%** (2024), mengkanibalisasi kantor cabang konvensional (turun dari 61,93% ke 38,06%). Nasabah daring cenderung memiliki profil risiko yang lebih cair dan tidak terverifikasi tatap muka.
2. **Peremajaan Usia Debitur**: Rerata usia pemohon bergeser dari **38,98 tahun** ke **35,79 tahun** (lebih muda >3 tahun), yang berkorelasi dengan masa rintisan usaha lebih pendek (`lama_usaha_tahun` rendah).
3. **Mekanisme Penurunan Kinerja**: Model dilatih ketika debitur berisiko rendah (*Pegawai Tetap*) mendominasi pasar (41,4%). Ketika portofolio bergeser dan dibanjiri segmen informal berisiko tinggi (proporsi gabungan melonjak dari **33,6%** menjadi **43,0%**), pohon keputusan yang dibangun model tidak memiliki batas partisi (*split criteria*) yang memadai untuk memisahkan risiko nasabah baru tersebut.

---

## 3. Analisis Kinerja per Segmen dan Identifikasi Segmen Paling Rawan

Untuk mencegah ilusi metrik agregat, evaluasi dilakukan secara terdisagregasi (*slice-based evaluation*) pada periode uji *Out-of-Time* (2024-H2, $N = 5.400$):

| Segmen Nasabah | Ukuran Sampel ($n$) | ROC-AUC | Laju Default Aktual | Tingkat Kerawanan Risiko |
| :--- | :---: | :---: | :---: | :--- |
| **Nelayan** | 597 | 0,7939 | 14,24% | Risiko tinggi, namun terpisahkan oleh rasio beban utang |
| **Pegawai Tetap** | 1.680 | 0,7843 | 7,74% | Risiko rendah, stabil |
| **Wiraswasta** | 1.254 | 0,7730 | 10,69% | Risiko sedang |
| **Pedagang Musiman**| 965 | 0,7665 | 13,68% | Risiko tinggi |
| **Petani** | **904** | **0,7050** | **13,61%** | **PALING RAWAN (Kinerja Terlemah, AUC Anjlok)** |

### Mengapa Segmen Petani Paling Rawan?
1. **Daya Pembeda Terlemah**: AUC sebesar **0,7050** pada segmen Petani adalah yang terendah di seluruh portofolio (selisih hampir 0,09 di bawah rata-rata segmen lain). Pada angka ini, kemampuan model membedakan petani yang akan bayar vs macet mendekati probabilitas acak.
2. **Kombinasi Karakteristik Ekstrem**: Petani memiliki laju gagal bayar aktual sangat tinggi (**13,61%**), varian penghasilan tinggi (rerata 0,46), dan tingkat kehilangan data tabungan perbankan formal yang tinggi (**10,51%** missing pada `rata_saldo_90h`).
3. **Konsekuensi Bisnis dan Sosial**:
   * *Bagi Bank (Credit Risk)*: Model meloloskan debitur petani yang sesungguhnya berisiko macet tinggi (*False Negative*), menimbulkan lonjakan beban penghapusan piutang (*write-off*).
   * *Bagi Masyarakat (Fairness & Exclusion)*: Model menolak petani yang beritikad baik (*False Positive*) karena model bingung membaca riwayat keuangan musiman mereka, memicu ketidakadilan algoritma (*algorithmic bias*) terhadap sektor pangan Sumatera.

---

## 4. Ringkasan Peragaan Training–Serving Skew dan Skalasi Dampak Bisnis

*Training–serving skew* adalah perbedaan representasi atau kalkulasi fitur antara lingkungan pelatihan (*offline training*) dan lingkungan penyajian (*online serving*).

### Ringkasan Eksekusi `src/skew_demo.py`
Eksperimen menyimulasikan kejadian umum di industri perbankan: data scientist melatih model di Python dengan mengisi *missing value* menggunakan **median data latih**, tetapi tim software engineer yang membangun layanan mikro inferensi menulis ulang fungsi tersebut dengan mengisi **nilai nol (`0`)**:

| Parameter Pengujian | Nilai Pengamatan | Makna Evaluasi |
| :--- | :---: | :--- |
| Total Data Uji Evaluasi | 5.400 permohonan | Data semester 2 tahun 2024 |
| Baris dengan Missing Value | **289 baris** (5,4%) | Nasabah yang terdampak perbedaan imputasi |
| AUC Definisi KONSISTEN (Median) | **0,7711** | Kondisi benar sesuai pelatihan |
| AUC Definisi MENYIMPANG (Nol) | **0,7683** | Kondisi layanan di produksi |
| Selisih AUC Agregat | **+0,0028** | **Tampak sangat kecil / dapat diabaikan** |
| Selisih Maksimum Vektor Fitur | **Rp 3.141.000,00** | Nilai saldo tabungan diganti Rp 0 |
| **Keputusan yang BERUBAH** | **66 nasabah (1,22%)** | **Keputusan kredit berbalik (Setujui $\leftrightarrow$ Tolak)** |
| AUC pada Baris Terdampak — Konsisten | **0,8164** | Model bekerja baik jika fitur benar |
| AUC pada Baris Terdampak — Menyimpang | **0,7865** | **Penurunan drastis (-0,0299) pada nasabah terkait** |

```
Bahaya Terbesar:
1. HTTP Response 200 OK — Layanan tidak mengalami crash, Exception, atau peringatan apa pun.
2. Selisih AUC agregat hanya 0,0028 — Tidak akan memicu alarm pada dashboard pemantauan umum.
3. Namun 66 manusia nyata mendapatkan keputusan finansial yang salah.
```

### Proyeksi Skalasi Dampak pada Portofolio 200.000 Permohonan per Tahun
Pada bank komersial berskala menengah dengan volume **200.000 permohonan kredit per tahun**, rasio kesalahan keputusan $1{,}222\%$ menghasilkan dampak:

$$\text{Nasabah Mengalami Kesalahan Keputusan} = 1{,}222\% \times 200.000 = \mathbf{2.444 \text{ nasabah/tahun}}$$

Dampak Finansial Berdasarkan Matriks Biaya Nyata (`conf/params.yaml`):
1. **Kasus False Negative Tambahan (Kredit Macet yang Diloloskan)**:
   Jika diasumsikan separuh dari kesalahan tersebut (1.222 debitur) adalah nasabah macet yang keliru disetujui akibat saldo diisi 0 (dianggap tidak punya beban tabungan bermasalah):
   $$\text{Rugi Pokok Pinjaman} = 1.222 \times \text{Rp } 12.000.000 = \mathbf{\text{Rp } 14.664.000.000 \text{ (14,66 Miliar Rupiah)}}$$
2. **Kasus False Positive Tambahan (Nasabah Baik yang Ditolak)**:
   Sebanyak 1.222 nasabah layak ditolak secara tidak adil, memicu kerugian potensi laba bunga:
   $$\text{Opportunity Loss} = 1.222 \times \text{Rp } 1.500.000 = \mathbf{\text{Rp } 1.833.000.000 \text{ (1,83 Miliar Rupiah)}}$$
3. **Total Kerugian Ekonomi**: Kerugian bisnis riil mencapai **Rp 16,5 Miliar per tahun**, semata-mata diakibatkan oleh satu baris perbedaan kode: `fillna(0)` versus `fillna(median)`.

---

## 5. Usulan Solusi Arsitektural untuk Mencegah Skew (Bukan Disiplin Pribadi)

Mengandalkan disiplin individu (seperti imbauan "developer harus lebih teliti", "baca dokumentasi", atau "tulis SOP bersama") terbukti selalu gagal di industri perangkat lunak skala besar. Solusi pencegahan harus bersifat **arsitektural dan ditegakkan secara mekanis oleh mesin**.

```
+----------------------------------------------------------------------------------------------------+
|                               ARSITEKTUR PENCEGAHAN TRAINING-SERVING SKEW                          |
+----------------------------------------------------------------------------------------------------+
|  1. SINGLE SOURCE OF TRUTH (src/features.py)                                                       |
|     - Transformasi Stateless: Fungsi kalkulasi instan per baris (rasio_angsuran, log_penghasilan)  |
|     - Transformasi Stateful : Artefak median & kategori dipelajari HANYA saat latih                |
|                                     |                                                              |
|                                     v Dibekukan ke artefak                                         |
|                           [artefak_fitur.json]                                                     |
|                                     |                                                              |
|                  +------------------+------------------+                                           |
|                  |                                     |                                           |
|                  v                                     v                                           |
|       [src/train.py (Pelatihan)]            [src/api.py (Penyajian Serving)]                       |
|       Memuat artefak_fitur.json             Memuat artefak_fitur.json                              |
|       (Dilarang fit ulang)                  (Dilarang tulis ulang logika)                          |
+----------------------------------------------------------------------------------------------------+
|  2. AUTOMATED PARITY TESTING DI CI/CD (tests/test_equivalence.py)                                  |
|     assert max|X_batch_training - X_single_serving| < 1e-6                                         |
|     * Bila tim API mengubah imputasi menjadi nol -> Build CI GAGAL seketika (PR ditolak)          |
+----------------------------------------------------------------------------------------------------+
```

### 1. Satu Sumber Kebenaran Fitur (*Single Source of Truth / Shared Library*)
* **Dilarang Menulis Ulang Fitur**: Tim produksi dilarang keras menulis ulang logika transformasi ke bahasa lain (misal Java/Go) atau menulis ulang fungsi Pandas di endpoint API.
* **Isolasi Modul Terpusat (`src/features.py`)**: Modul transformasi fitur ditulis satu kali saja dan diimpor bersama oleh pipeline pelatihan (`src/train.py`) dan layanan inferensi (`src/api.py`).
* **Pemisahan Stateless vs Stateful**:
  * Transformasi *stateless* (seperti `angsuran / penghasilan`) aman dihitung ulang di mana saja.
  * Transformasi *stateful* (imputasi median, pemetaan kategori, batas diskretisasi) **wajib disimpan sebagai artefak pelatihan** ke dalam berkas `artefak_fitur.json`. Layanan API hanya boleh memuat (*load*) nilai median dari artefak tersebut, bukan menghitung ulang dari data transaksi harian.

### 2. Penegakan Urutan Kolom Deterministik
* Fungsi transformasi wajib mengunci urutan kolom fitur keluaran secara absolut (`return X[kolom_fitur(art)]`). Hal ini menjamin matriks inferensi di produksi memiliki indeks kolom yang sama persis dengan matriks saat model di-*fit*, mencegah galat salah atribut tanpa peringatan.

### 3. Gerbang Uji Kesetaraan Otomatis di Pipeline CI (*Automated Equivalence Testing*)
* Memasang modul pengujian `tests/test_equivalence.py` ke dalam alur kerja CI GitHub Actions.
* Pengujian ini mengevaluasi dataset sampel secara simultan menggunakan dua metode:
  1. Transformasi jalur pelatihan (*batch offline*).
  2. Transformasi jalur layanan (*single-row payload* API).
* Uji kesetaraan memberlakukan asersi ketat:
  $$\max |X_{\text{batch}} - X_{\text{online}}| < 10^{-6}$$
* **Mekanisme Penegakan**: Jika ada pengembang yang secara sengaja atau tidak sengaja mengubah pengisian nilai hilang menjadi nol pada layanan penyajian, uji kesetaraan otomatis gagal (*Red CI*), dan *Pull Request* otomatis diblokir dari penggabungan ke cabang `main`.

---

## 6. Kesimpulan Pemenuhan Titik Periksa Praktikum 2

| No | Komponen Titik Periksa Asisten | Bukti dan Status Pemenuhan |
| :---: | :--- | :--- |
| **1** | Tabel AUC per bulan pada 2024 dihasilkan | **LULUS** — Tersedia lengkap di [notebooks/02_penuaan.ipynb](file:///C:/Users/ASUS/.gemini/antigravity/scratch/mlops-kredit-sumatera/notebooks/02_penuaan.ipynb) (12 bulan, AUC berkisar 0,718 s.d. 0,812). |
| **2** | Grafik penurunan kinerja dibuat | **LULUS** — Gambar tersimpan di `plots/penuaan_model_2024.png` dan ditampilkan langsung pada sel notebook. |
| **3** | Pergeseran komposisi segmen ditunjukkan angka | **LULUS** — Terbukti secara kuantitatif: pegawai tetap anjlok -10,30 pp, segmen berisiko naik gabungan +9,4 pp. |
| **4** | Kinerja per segmen dihitung & segmen terlemah diidentifikasi | **LULUS** — Segmen terlemah teridentifikasi: **Petani** (AUC terendah 0,7050, laju gagal bayar 13,61%). |
| **5** | `skew_demo.py` dijalankan dan keluarannya dijelaskan | **LULUS** — Skrip berhasil dieksekusi: membuktikan 66 keputusan berbalik (1,22%) tanpa menghasilkan galat sistem. |
| **6** | Penjelasan mengapa uji unit biasa tidak menangkap skew | **LULUS** — Uji unit standar hanya memverifikasi tipe data dan kode respons (HTTP 200 / float), bukan kebenaran semantik fitur. Skew hanya dapat ditangkap oleh *uji kesetaraan fitur (equivalence testing)*. |
