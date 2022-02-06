# Laporan Proyek Machine Learning - Rifqi Arrahim
 
## Project Overview
Anime merupakan salah satu hiburan yang banyak digemari orang orang. Menurut dokumen tahun 2004 dari Japan External Trade Organization dari artikel [12 SHOCKING FACTS ABOUT ANIME YOU NEED TO KNOW](https://newslanded.com/2020/07/03/12-shocking-facts-about-anime-you-need-to-know/), film anime dan acara televisi menyumbang 60% dari hiburan berbasis animasi dunia. Hampir 40 sekolah di Jepang telah mendeklarasikan anime sebagai mata pelajaran tersendiri. Akting suara anime juga sangat besar, karena Jepang memiliki sekitar 130 sekolah akting suara.
## Business Understanding
Setiap Anime memiliki genre, tipe dan jumlah episode yang berbeda-beda. Berikut adalah tipe-tipe anime:
1. Movie
2. TV
3. OVA(Original Video Animation)
4. Music
5. ONA(Original Net Anime)
6. Special
 
### Problem Statements
- Bagaimana membuat sistem rekomendasi anime berdsarkan tipe anime?
- Dengan data rating yang ada, bagaimana memberikan rekomendasi anime yang mungkin akan disukai penonton?
### Goals
- Membuat sistem rekomendasi anime berdasarkan tipe anime.
- Memberikan rekomendasi anime yang mungkin akan disukai penonton berdsarkan rating yang diberikan.
 
### Solution Approach
- Membuat sistem rekomendasi yang telah dipersonalisasi menggunakan teknik content based filtering
- Memberikan sejumlah rekomendasi animme yang sesuai dengan preferensi penonton dengan teknik collaborative filtering
 
## Data Understanding
Data Understanding adalah tahap awal proyek untuk memahami data yang dimiliki. Tahap Data Understanding penting untuk memahami variabel-variabel pada data serta korelasi antar variabel. Dalam kasus ini, kita memiliki 2 [file](https://www.kaggle.com/CooperUnion/anime-recommendations-database) terpisah mengenain anime dan rating. Pada file anime terdapat 12294 record dan 6 fitur yaitu anime_id, name, genre, type, episodes, rating, dan members. Pada file rating terdapat 7813737 record dan 3 fitur yaitu user_id, anime_id, dan rating.
### Anime.csv
1. anime_id - myanimelist.net's unique id identifying an anime.
2. name - full name of anime.
3. genre - comma separated list of genres for this anime.
4.type - movie, TV, OVA, etc.
5. episodes - how many episodes in this show. (1 if movie).
6. rating - average rating out of 10 for this anime.
7. members - number of community members that are in this anime's "group".
### Rating.csv
1. user_id - non identifiable randomly generated user id.
2. anime_id - the anime that this user has rated.
3. rating - rating out of 10 this user has assigned (-1 if the user watched it but didn't assign a rating).
<br>



![infoanime](infoanime.jpg)<br>
Berdasarkan output diatas, kita dapat mengetahui anime.csv memiliki 12294 entri
<br>
![tipeanime](tipeanime.jpg)<br>
Terdapat 12294 data anime yang unik dengan 6 jenis tipe anime.<br>
![describerating](describerating.jpg)<br>
Dari output diatas, diketahui bahwa nilai maksimum rating adalah 10 dan nilai minimum -1.<br>
![inforating](ratinganime.jpg)<br>
Berdasarkan output diatas, kita dapa mengetahui jumlah user yang memberikan rating, jumlah anime, dan jumlah data rating.

 
 
## Data preparation
### Missing Value
Saya mengecek nilai 0 pada file anime dan rating. Ketika ada data yang bernilai 0, Saya akan menghapus record tersebut. Hal ini perlu dilakukan untuk menghindari membuat model machine learning yang bias.
### Feature Engineering
Feature Engineering merupakan proses membuat variabel input baru dari variabel data yang sudah ada. 
- Penulis melakukan konversi data series anime_id, name, type menjadi list. Dalam hal ini, kita menggunakan fungsi tolist() dari library numpy.
- Penulis membuat dictionary untuk menentukan pasangan key-value pada data id, name, dan type.
- Pada data rating terdapat nilai -1 yang menandakan user belum memberi rating. Penulis mengubah -1 menjadi 0 karena akan mempengaruhi model.
- Penulis melakukan encode fitur ‘user_id’ dan ‘anime_id’ ke dalam indeks integer.
- Penulis mengubah tipe data fitur rating menjadi float.
- Rating memiliki data berjumlah 7813737. Penulis memutuskan mengambil sample sebanyak 73515 sesuai dengan jumlah user. Hal ini dilakukan karena keterbatasan runtime pada Google Colab. Untuk mengolah data sebanyak itu penulis perlu menggunakan Google Colab Pro
### Data Transform
- Penulis membagi dataset menjadi data latih sebanyak 80% dan data validasi sebanyak 20%.
## Modelling
### Content Based Filtering
Penulis membangun sistem rekomendasi sederhana berdasarkan tipe anime. 
#### TF-IDF Vectorizer
Teknik TF-IDF Vectorizer akan digunakan pada sistem rekomendasi untuk menemukan representasi fitur penting dari setiap tipe anime. Penulis menggunakan fungsi tfidfvectorizer() dari library sklearn.<br>
![matriks](matriksanime.jpg)<br>
Berikut vektor tf-idf dalam bentuk matriks. Output matriks tf-idf di atas menunjukkan Little Polar Bear: Shirokuma-kun, Fune ni Noru memiliki kategori OVA. Hal ini terlihat dari nilai matriks 1.0 pada kategori OVA.
#### Cosine Similarity
Pada tahap sebelumnya, penulis telah berhasil mengidentifikasi korelasi antara anime dengan tipenya. Sekarang, penulis akan menghitung derajat kesamaan (similarity degree) antar anime dengan teknik cosine similarity. Di sini, penulis menggunakan fungsi cosine_similarity dari library sklearn. <br>
![similarity](similarity.jpg)<br>

### Collaborative Filtering

## Evaluation
### Content Based Filtering

### Collaborative Filtering
 
