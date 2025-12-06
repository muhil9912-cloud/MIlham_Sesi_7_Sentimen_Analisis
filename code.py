# Kompilasi Materi Sesi 7: Analisis Sentimen (BoW & TF-IDF)
# Nama: Heriswaya
# Tugas: Sesi 7 PRIMA MAGANG PTKI
# Notebook ini adalah gabungan dari semua materi dan kode latihan Sesi 7.
# Praktik dua metode: Bag-of-Words (BoW) dan TF-IDF untuk klasifikasi sentimen ulasan film.

# Bagian 1: Analisis Sentimen dengan Bag-of-Words (BoW)
# Metode BoW mengubah teks menjadi angka berdasarkan frekuensi kemunculan kata.

# 1.1 Import Library (BoW)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("Library untuk BoW siap.")

# 1.2 Menyiapkan Data Training (BoW)
# Data seimbang: 3 positif, 3 negatif
corpus_bow = [
    'Saya suka film ini',
    'film ini sangat bagus dan saya suka',
    'review film ini luar biasa bagus',
    'Saya benci film itu',
    'Saya tidak suka film ini',
    'film ini jelek sekali'
]

y_labels_bow = [
    'Positif', 'Positif', 'Positif',
    'Negatif', 'Negatif', 'Negatif'
]

print(f"Data BoW: {len(corpus_bow)} ulasan, {len(y_labels_bow)} label.")

# 1.3 Vektorisasi (BoW) dengan N-grams
# Menggunakan ngram_range=(1, 2) untuk unigram dan bigram
vectorizer_bow = CountVectorizer(ngram_range=(1, 2))
X_bow = vectorizer_bow.fit_transform(corpus_bow)

print("Vektorisasi BoW (dengan n-gram) selesai.")
# Opsional: print(vectorizer_bow.get_feature_names_out())

# 1.4 Melatih Model (BoW)
model_bow = MultinomialNB()
model_bow.fit(X_bow, y_labels_bow)

print("Model BoW berhasil dilatih!")
print("=" * 30)

# 1.5 Uji Coba Model (BoW)
review_baru_bow = [
    'Saya benci film ini',  # Harusnya 'Negatif'
    'film ini bagus dan saya suka'  # Harusnya 'Positif'
]

X_baru_bow = vectorizer_bow.transform(review_baru_bow)
prediksi_bow = model_bow.predict(X_baru_bow)

print("Hasil Prediksi Model BoW:")
for i, review in enumerate(review_baru_bow):
    print(f"Review: {review} -> Prediksi: {prediksi_bow[i]}")

# Bagian 2: Analisis Sentimen dengan TF-IDF
# TF-IDF mengukur kepentingan kata relatif terhadap corpus.

# 2.1 Import Library (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

print("\nLibrary untuk TF-IDF siap.")

# 2.2 Menyiapkan Data Training (TF-IDF)
corpus_tfidf = [
    'Saya suka film ini',
    'Saya benci film itu',
    'film ini sangat bagus dan saya suka'
]
y_labels_tfidf = ['Positif', 'Negatif', 'Positif']

print(f"Data TF-IDF: {len(corpus_tfidf)} ulasan, {len(y_labels_tfidf)} label.")

# 2.3 Vektorisasi (TF-IDF) dengan Stopwords
# Mengabaikan stopwords bahasa Indonesia
tfidf_vectorizer = TfidfVectorizer(stop_words=['saya', 'ini', 'itu', 'dan'])
X_tfidf = tfidf_vectorizer.fit_transform(corpus_tfidf)

print("Vektorisasi TF-IDF (dengan stop_words) selesai.")
# Opsional: print(tfidf_vectorizer.get_feature_names_out())

# 2.4 Melatih Model (TF-IDF)
model_tfidf = MultinomialNB()
model_tfidf.fit(X_tfidf, y_labels_tfidf)

print("Model TF-IDF berhasil dilatih!")
print("=" * 30)

# 2.5 Uji Coba Model (TF-IDF)
review_baru_tfidf = [
    'Saya benci film ini',  # Harusnya 'Negatif'
    'film ini bagus dan saya suka'  # Harusnya 'Positif'
]

X_baru_tfidf = tfidf_vectorizer.transform(review_baru_tfidf)
prediksi_tfidf = model_tfidf.predict(X_baru_tfidf)

print("Hasil Prediksi Model TF-IDF:")
for i, review in enumerate(review_baru_tfidf):
    print(f"Review: {review} -> Prediksi: {prediksi_tfidf[i]}")

# Bagian 3: Kesimpulan
# Kedua model (BoW dan TF-IDF) berhasil dikembangkan untuk analisis sentimen.
# BoW efektif dengan n-grams untuk konteks, TF-IDF untuk bobot kata penting.
# Kedua teknik dasar kuat untuk klasifikasi teks NLP.
