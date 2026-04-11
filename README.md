# Fourier Lab

## Interaktivna analiza, rekonstrukcija i skrivanje informacija u slikama

---

## 📌 Opis projekta

Fourier Lab je interaktivna aplikacija za obradu slika koja koristi Furijeovu transformaciju za analizu i manipulaciju slika u frekvencijskom domenu.

Cilj projekta je da pokaže kako se slike mogu:

* analizirati kroz frekvencije
* rekonstruisati iz frekvencijskih komponenti
* filtrirati (uklanjanje šuma, oštrenje)
* koristiti za skrivanje informacija (steganografija)

Aplikacija omogućava rad u realnom vremenu kroz grafički interfejs.

---

## 🚀 Funkcionalnosti

### 1. Frekvencijska analiza

* Upload slike
* Prikaz originalne slike
* Prikaz FFT spektra (log magnitude)

### 2. Rekonstrukcija slike

* Postepena rekonstrukcija slike iz frekvencija
* Slider za kontrolu broja korišćenih frekvencija
* Vizualizacija kako se slika "gradi"

### 3. Filtriranje

* Low-pass filter (zamućenje)
* High-pass filter (oštrenje)
* Ručno uklanjanje frekvencija (interaktivno)

### 4. Steganografija

* Ugradnja skrivene poruke u sliku
* Ekstrakcija skrivene poruke
* Kontrola intenziteta skrivenih podataka

### 5. Eksperimenti

* Analiza uticaja frekvencija na kvalitet slike
* Merenje kvaliteta (MSE, PSNR)

---

## 🛠️ Tehnologije

* Python
* NumPy
* OpenCV
* Streamlit
* Matplotlib (opciono)

---

## 📂 Struktura projekta

```
fourier_lab/
│
├── app.py                # Glavna aplikacija (Streamlit)
├── fft_utils.py         # Funkcije za FFT
├── filters.py           # Filtri u frekvencijskom domenu
├── reconstruction.py    # Rekonstrukcija slike
├── steganography.py     # Skriveni podaci u slici
└── utils.py             # Pomoćne funkcije
```

---

## ⚙️ Instalacija

1. Kloniraj repozitorijum:

```
git clone <repo_link>
cd fourier_lab
```

2. Instaliraj zavisnosti:

```
pip install numpy opencv-python streamlit matplotlib
```

---

## ▶️ Pokretanje aplikacije

Pokreni aplikaciju pomoću Streamlit-a:

```
streamlit run app.py
```

Aplikacija će se otvoriti u browseru.

---

## 🧠 Teorijska osnova

Projekat se zasniva na Furijeovoj transformaciji koja omogućava predstavljanje slike kao sume frekvencijskih komponenti.

* Niske frekvencije → globalna struktura slike
* Visoke frekvencije → detalji i ivice

Manipulacijom ovih komponenti moguće je menjati izgled slike.

---

## 🎯 Cilj projekta

* Razumevanje Furijeove transformacije kroz vizuelizaciju
* Primena u realnim problemima obrade slika
* Razvoj interaktivnog alata za eksperimentisanje

---

## 📈 Moguća proširenja

* Obrada video signala
* 3D vizualizacija spektra
* Poređenje sa DCT (JPEG kompresija)
* Rad u realnom vremenu sa kamerom

---

## 👨‍💻 Autor

[Dodaj svoje ime]

---

## 📜 Licenca

Ovaj projekat je razvijen u edukativne svrhe.
