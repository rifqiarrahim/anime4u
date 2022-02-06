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
 
## Modelling
Tahap ini saya mengembangkan model machine learning dengan tiga algoritma yaitu KNN, Decision Tree, dan SVM. Model dibangung dengan bantuan library skicit learn. Setelah memanggil library penulis melakukan hyperparameter tuning dengan bantuan gridsearchcv. <br>
### SVM
Model ini membuat sebuah Hyperplane di antara data dengan margin yang maksimal. Hyperplane adalah sebuah garis yang memisahkan data positif diabetes dan negatif diabetes. Margin adalah jarak Hyperplane dengan data tersebut. Semakin besar margin yang dibuat, semakin tinggi akurasi yang kita dapatkan.
- C : 1.0<br>
Fungsi Parameter : Seberapa besar kita ingin menghindari kesalahan klasifikasi setiap training. Saya menggunakan nilai default yaitu 1.0.
- degree : 3<br>
Fungsi Parameter : Derajat fungsi kernel. Saya menggunakan nilai default yaitu 3.
- gamma : scale<br>
Fungsi Parameter : Seberapa jauh pengaruh setiap satu training. Saya menggunakan nilai default yaitu 'scale'.
### KNN
Pertama model akan menentukan nilai k. Nilai k adalah jumlah n_neighbors atau data terdekat yang akan dijadikan acuan. Lalu model akan menghitung jarak tersebut dan melakukan klasifikasi.
- leaf_size : 30<br>
Fungsi Parameter : leaf_size dapat mempengaruhi kecepatan dan memori yang diperlukan untuk menyimpan pohon.
- n_neighbors : 10<br>
Fungsi Parameter : n_neighbors untuk mengatur nilai k pada KNN.
- weights : distance<br>
Fungsi Parameter : Semakin dekat data maka akan semakin besar pengaruh yang diberikan.
- metric : euclidian<br>
Fungsi Parameter : Fungsi untuk menghitung jarak
### Decision Tree
Model akan membuat sebuah root nodes, lalu akan membuat percabangan dengan memilih fitur-fitur yang ada dengan menghitung entropy dan information gain. Ketika tree sudah mencapai leaf nodes atau tidak mempunyai cabang lagi, maka nodes tersebut merupakan output dari model
- criterion : gini<br>
Fungsi Parameter : Fungsi untuk mengukur kualitas percabangan.
- max_depth : 3<br>
Fungsi Parameter : Maksimum level pohon.
- max_features : None<br>
Fungsi Parameter : Menentukan jumlah fitur ketika bercabang.
- min_samples_leaf : 5<br>
Fungsi Parameter : Minimum jumlah sampel yang dibutuhkan untuk leaf node.
- min_samples_split : 2<br>
Fungsi Parameter : Minimum jumlah sampel untuk internal node bercabang.
 
## Evaluation
Fitur glucose memiliki nilai korelasi yang besar dengan label outcome berdasarkan hal itu dapat disimpulkan bahwa kadar glukosa dalam tubuh merupakan faktor yang paling berpengaruh untuk mengidentifikasi apakah pasien mengidap diabetes atau tidak.<br>
Dari ketiga model di atas saya menggunakan metrics.classisfication_report yang merupakan library bawaan dari skicitlearn untuk mengevaluasi model. Output dari model dapat diklasifikasikan menjadi empat.<br>
1. True Positive atau TP. Ketika model memberikan output positif diabetes dan label data test bernilai positif diabetes.
2. True Negative atau TN. Ketika model memberikan output negatif diabetes dan label data test bernilai negatif diabetes.
3. False Positive atau FP. Ketika model memberikan output positif diabetes sedangkan label data test bernilai negatif diabetes.
4. False Negative atau FN. Ketika model memberikan output negatif diabetes sedangkan label data test bernilai positive diabetes.
$$Precision = \frac{TP}{TP + FP} $$<br>
$$Recall = \frac{TP}{TP + FN} $$<br>
$$F1 = \frac{2 * Precision*Recall}{Precision+Recall}$$<br>
### SVM
![SVM](SVM(2).jpg)
### KNN
![KNN](KNN(2).jpg)
### Decision Tree
![DCT](DCT(2).jpg)<br>
Nilai precision, recall, dan f1-score sering digunakan untuk mengevaluasi masalah klasifikasi. Dari nilai f1-score model KNN dan Decision Tree memiliki nilai lebih baik dibandingkan model SVM. Jadi dapat disimpulkan model KNN dan Decision Tree dapat memberikan output yang lebih tepat dibandingkan model SVM. 
 
