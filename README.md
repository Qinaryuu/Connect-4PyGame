# Connect-4PyGame 🎮🔴🟡

**Pengembangan Agen AI pada Permainan Connect Four Menggunakan Kombinasi Adversarial Search, Forward Chaining, dan Penalaran Bayesian**

Proyek Tugas Besar mata kuliah **Pengantar Kecerdasan Artifisial** yang mengimplementasikan permainan papan strategi Connect Four (7x6) berbasis Python dan Pygame, dilengkapi dengan agen AI hibrida serta platform pengujian performa algoritma secara terukur.

---

## 👥 Anggota Kelompok 5

| No | Nama Lengkap | NIM |
| :--- | :--- | :--- |
| 1 | Muhammad Rahman Fakhrichi | 11251018 |
| 2 | Niken Qinata | 11251043 |
| 3 | Evi Juarsa | 11251045 |
| 4 | Kirana Muthmainnah | 11251071 |
| 5 | M. Luthfi Handayudha | 11251089 |

---

## 🧠 Arsitektur Kecerdasan Artifisial

Sistem ini mengintegrasikan tiga paradigma kecerdasan artifisial dalam **Agen Utama (A5)** untuk menghasilkan keputusan yang cepat, taktis, dan adaptif:

1. **Penalaran Probabilistik Bayesian (*Opponent Modeling*)**
   Memperbarui probabilitas gaya bermain lawan (agresif vs. defensif) secara *real-time* menggunakan Teorema Bayes berdasarkan langkah terakhir lawan, lalu mengkalibrasi bobot serangan dan pertahanan pada fungsi heuristik secara dinamis.
2. **Agen Logika (*Forward Chaining*)**
   Mengevaluasi kondisi papan secara langsung untuk mengeksekusi aturan kritis seperti **Instant Win** (menang dalam 1 langkah) dan **Instant Block** (memblokir ancaman 3-sebaris lawan) tanpa perlu melakukan penelusuran pohon keputusan.
3. **Adversarial Search (*Minimax + Alpha-Beta Pruning + Move Ordering*)**
   Digunakan pada situasi non-kritis dengan memprioritaskan evaluasi pada lajur tengah (*move ordering*) untuk memaksimalkan pemangkasan cabang pohon pencarian secara efisien.

### Agen Pembanding (*Baseline*)
Untuk mengevaluasi efektivitas Agen Utama (A5), proyek ini juga menyediakan empat agen pembanding:
* **Random Agent:** Memilih kolom valid secara acak.
* **Pure Rule-Based Agent:** Hanya mengandalkan aturan *Forward Chaining*.
* **Breadth-First Search (BFS) Agent:** Pencarian tanpa informasi tingkat demi tingkat.
* **Standard Minimax Agent:** Algoritma Minimax tanpa optimasi *Alpha-Beta Pruning*.

---

## 📂 Struktur Proyek

```text
Connect-4PyGame/
├── main.py             # Entry point aplikasi & antarmuka grafis (Pygame)
├── board.py            # Pemodelan matriks 7x6, validasi langkah, & Goal Test
├── ai_agent.py         # Implementasi Agen Utama A5 (Bayesian + FC + Alpha-Beta)
├── baselines.py        # Implementasi agen pembanding (Random, Rule-Based, BFS, Minimax)
├── benchmark.py        # Skrip simulasi otomatis & pencatatan metrik performa
└── README.md           # Deskripsi proyek