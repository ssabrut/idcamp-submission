# Laporan Proyek Machine Learning - Michael Eko

## Domain Proyek

### Latar Belakang
Industri perfilman saat ini sudah menjadi industri yang terbilang cukup besar, dimana angka penonton bioskop terus meningkat dari tahun ke tahun. Hal tersebut dilihat dari tahun 2018 yang jumlah penonton bioskop di Indonesia telah mencapai 50 juta penonton dengan jumlah produksi film luar negeri hingga dalam negeri sebanyak hampir 200 judul film yang telah tayang di seluruh Indonesia (Tren Positif Film Indonesia | Indonesia.go.id, 2019). Tidak hanya dari segi plot cerita, film saat ini menghadirkan visual yang membuat penonton terkagum-kagum. Terutama saat pandemi seperti ini, menonton film merupakan kegiatan alternatif yang memberikan hiburan untuk mengusir kebosanan.

Tak jarang juga orang menonton film memang karena hobi. Hal ini menyebabkan industri perfilman harus menghadapi persaingan yang ketat dalam menciptakan terobosan baru guna memenuhi kebutuhan konsumen yang kian beragam [1]. Sudah banyak platform yang memberi fasilitas untuk menonton film seperti Netflix, Youtube, Viu dan sebagainya. Dari sekian banyaknya film yang terdapat pada platform tersebut, membuat calon penonton kesulitan dalam menentukan film yang akan ditontonnya. Untuk mencari film yang akan ditonton saja akan memakan waktu yang cukup lama, selain itu film yang sudah ditentukan belum tentu sesuai dengan keiinginan calon penonton.

Selera setiap orang juga menjadi suatu hal yang perlu dipertimbangkan karena setiap orang pasti memiliki selera yang berbeda [2]. Seseorang bisa saja menyukai film berdasarkan genre, aktor/aktris, atau bahkan rumah produksinya sendiri. Hal ini bisa menjadi permasalahan dalam menentukan film yang sesuai dengan ekspektasi. Mengingat jumlah film yang begitu banyak dan tersebar di berbagai platform.

## Business Understanding

### Problem Statement
Berdasarkan pada latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah bagaimana cara  membangun suatu sistem rekomendasi film menggunakan pendekatan *Content Based Filtering* dan *Collaborative User Filtering* guna mempermudah *user* dalam menentukan preferensi film yang ingin ditontonnya.

### Goals
Tujuan proyek ini dibuat adalah membangun sebuah sistem rekomendasi untuk mempermudah *user* dalam menentukan preferensi film mereka.

### Solution Statement
Solusi yang dapat dilakukan agar goals terpenuhi adalah membangun sebuah sistem rekomendasi untuk memberikan rekomendasi film berdasarkan keterkaitan sebuah film dan preferesnsi *user* dengan pendekatan *Content Based Filtering* dan *User Collaborative Filtering*.

## Data Understanding
---

Dataset yang digunakan pada proyek ini diambil dari website kaggle [*The Movies Dataset*](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). Dataset yang digunakan pada proyek ini ada 3, yaitu *movies_metadata.csv, links_small.csv* dan *ratings_small.csv*. Semua dataset memiliki format *.csv* dengan informasi masing-masing sebagai berikut :
* *movies_metadata.csv* memiliki total 45,466 data dan 24 fitur :
    * adult : merupakan keterangan apakah film termasuk film dewasa atau tidak
    * belongs_to_collection : merupakan koleksi dari film
    * budget : merupakan anggaran yang dibutuhkan untuk membuat film
    * genres : merupakan genre dari film
    * homepage : merupakan link website dari film
    * id : merupakan id dari film
    * imdb_id : merupakan id dari film di website imdb
    * original_language : merupakan bahasa asli dari film
    * original_title : merupakan judul awal dari film
    * overview : merupakan gambaran singkat dari film
    * popularity : merupakan popularitas dari film
    * poster_path : merupakan link poster dari film
    * production_companies : merupakan perusahaan yang memproduksi film
    * production_countries : merupakan negara dimana film di buat
    * release_date : merupakan tanggal film dirilis
    * revenue : merupakan pendapatan dari flim
    * runtime : merupakan durasi dari film
    * spoken_languages : merupakan bahasa yang digunakan pada film
    * status : merupakan status dari film (sudah dirilis atau belum)
    * tagline : merupakan tagline dari film
    * title : merupakan judul film saat ini
    * video : apakah film memiliki video atau tidak
    * vote_average : merupakan rata2 vote film
    * vote_count : merupakan jumlah vote dari film


* *links_small.csv* memiliki total 9,125 data dan 3 fitur :
    * movieId : merupakan id dari film
    * imdbId : merupakan id dari film di website imdb
    * tmdbId : merupakan id dari film di website tmdb


* *ratings_small.csv* memiliki total 100,004 data dan 4 fitur :
    * userId : merupakan id dari user
    * movieId : merupakan id dari film
    * rating : merupakan rating yang diberikan user
    * timestamp : merupakan waktu kapan rating diberikan 

Berdasarkan seluruh dataset di atas, kita dapat membuat sebuah sistem rekomendasi dengan 2 pendekatan, yaitu *Conten Based Filtering* dan *Collaborative User Filtering*. Untuk pendekatan *Content Based* kita dapat menggunakan kesamaan deskripsi dari film, sedangkan *Collabrative User* kita dapat gunakan rating yang telah diberikan seorang user kepada film.

### Exploratory Data Analysis
Sebelum melakukan pemrosesan data, ada baiknya untuk mengeksplor data untuk mengetahui keadaan data seperti mencari *missing value*, menghapus fitur yang tidak relevan dan menentukan fitur yang akan digunakan.
* *Missing value in dataset*
     Ada baiknya kita mencari *missing value* pada setiap dataset dan menentukan apa yang harus dilakukan selanjutnya agar data tersebut optimal. Untuk *missing value* secara keseluruhan kita biarkan kosong, namun fitur yang ingin kita gunakan harus kita *fill missing value*nya. Seperti fitur *genre* terdapat *missing value*, kita dapat mengisi *missing value* dengan menggunakan [pandas.DataFrame.fillna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html) dengan value *list* kosong :
    ```
    metadatas['genres'] = metadatas['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    ```
    * *movies_metadata.csv* mempunyai total 105,562 data yang dimana hal tersebut terbilang cukup banyak dan dapat mempengaruhi performa model. Namun data yang hilang tidak terdapat pada fitur yang ingin kita gunakan maka dari itu kita dapat menghiraukan data yang hilang atau bisa kita hapus.
    * *links_small.csv* mempunyai total 13 data yang hilang. Hal ini tidak akan mempengaruhi model maka dari itu bisa kita hapus atau kita hiraukan.
    * *ratings_small.csv* tidak memiliki data yang hilang.

* Menghapus fitur yang tidak relevan
    Pada dataset *movies_metadata.csv* kita akan menghapus cukup banyak fitur yang dimana fitur tersebut tidak akan digunakan pada pembuatan sistem rekomendasi seperti *belongs_to_collection, homepage, imdb_id, overview, poster_path, tagline*.

* Menentukan fitur yang akan digunakan
    Karena pada proyek ini kita menggunakan 2 pendekatan, maka akan dijabarkan sebagai berikut :
    * *Content Based Filtering*
        Untuk *Content Based Filtering* kita akan memberikan rekomendasi film kepada *user* berdasarkan kemiripan dari deskripsi film. Jika suatu film memiliki deskripsi yang mirip dengan film lain, maka film tersebut akan direkomendasikan kepada *user*.

    * *User Collaborative Filtering*
        Untuk *User Collaborative Filtering* kita akan memberikan rekomendasi film kepada *user* jika *user* tersebut memiliki kemiripan preferensi dengan *user* lain. Sebagai contoh jika *user A* suka dengan genre *drama, comedy* dan *user B* suka dengan genre *comedy*, maka kemungkinan besar *user B* akan direkomendasikan film dengan genre drama juga.

# Data Preparation

### Menormalisasi fitur genre
Karena data dari fitur genre merupakan sebuah *list* yang berisikan *JSON Object*, maka kita perlu mengganti *JSON Object* ke sebuah *string* biasa. Sehingga data dari genre merupakan sebuah *list* yang berisikan genre.

### *Dropping invalid row*
Setelah melakukan eksplorasi pada dataset, didapatkan sebuah data yang tidak valid yang terletak pada index 19.730, 29.503, dan 35.587. Karena data yang tidak valid sangat sedikit maka kita dapat menghapus data tersebut.

### *Feature Engineering*
Proses ini hanya dilakukan untuk pendekatan *Content Based Filtering* dimana kita akan menggabungkan fitur *tagline* dengan *overview* untuk membuat sebuah fitur baru yaitu *description*.

# Modeling

Model yang akan digunakan pada proyek ini akan menggunakan 2 pendekatan yaitu *Content Based Filtering* dan *User Collaborative Filtering*. Dimana *Content Based Filtering* akan menggunakan *Cosine Similarity Score* yang menghitung kesamaan dari sebuah film dengan film lainnya. Lalu untuk *User Collaborative Filtering* akan menggunakan library khusus *recommendation system* yaitu [*Surprise*](https://surpriselib.com/) yaitu SVD (*Singular Value Decomposition*) untuk meminimalisir *RMSE*.

### *Content Based Filtering*
Content based filtering menggunakan informasi tentang beberapa item/data untuk merekomendasikan kepada pengguna sebagai referensi mengenai informasi yang digunakan sebelumnya. Tujuan dari content based filtering adalah untuk memprediksi persamaan sejumlah informasi yang didapat dari pengguna. Sebagai contoh, seorang penonton sedang menonton film tentang korupsi. Platform penyedia film online secara sistem akan merekomendasikan si pengguna untuk menonton film lain yang berhubungan dengan korupsi.

Dalam pembuatannya, *Content Based Filtering* menggunakan konsep perhitungan vektor, yaitu *TF-IDF* dan *Cosine Similarity Score* yang mengonversikan data/teks ke dalam bentuk vektor. *Cosine Similarity* adalah sebuah metric untuk mengukur kemiripan 2 *item*. Secara matematis, *Cosine Similarity* mengukur sudut cosinus antara 2 vektor yang diproyeksikan dalam ruang multidimensi.

*TF-IDF* umum digunakan pada korpus teks, pendekatan ini memiliki beberapa proses yang menarik yang akan berguna untuk mendapatkan representasi vektor dari data sehingga untuk pendekatan *Conten Based Filtering* kita dapat menggunakan *TF-IDF* berdasarkan genre dari film dengan rumus matematis sebagi berikut :

$$W_{i,j} = tf_{i,j}\cdot\log\tfrac{N}{df_{i}}$$

dimana :
* $$tf_{i,j}$$ : jumlah kemunculan i dan j
* $$df_{i,j}$$ : jumlah dokumen
* N : jumlah total dokumen

Setelah menghitung *TF-IDF* selanjutnya kita akan menghitung *Cosine Similarity Score* menggunakan library dari [from sklearn.metrics.pairwise.cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html). Dimana *Cosine Similarity* ini berfungsi untuk mengukur kemiripan antara 2 vektor tersebut yang merupakan ukuran seberapa mirip preferensi antara kedua orang tersebut. Hasil keluaran berkisar antara 0-1 dengan rumus matematis sebagai berikut :

cosine similarity = $$\cos\theta$$ = $$\tfrac{b . c}{\begin{Vmatrix} b \end{Vmatrix} \begin{Vmatrix} c \end{Vmatrix}}$$

dimana :
* b . c : dot product kedua vektor
* ||b|| ||c|| : dot product besaran vector

Sebagai contoh kita ingin mengecek apakah Johan dan Mega mempunyai kesamaan dalam preferensi film dan kita hanya memiliki dataset rating dari 2 film. Rating berkisar antara 1-5, dimana 5 adalah skor terbaik dan 1 adalah skor terburuk, dan 0 kalau orang tersebut belum menonton film tersebut.

| Name | Iron Man (2008) | Pride & Prejudice (2005 |
| ---- | ----------------| ------------------------|
| Johan | 4 | 3 |
| Mega | 5 | 5 |
Tabel 1, dataset *dummy* review dari film

Kita dapat mengubah tabel tersebut ke dalam bentuk vektor kolom yang terpisah sesuai dengan rating dari masing-masing *user*.
$$\overrightarrow{j} = \begin{bmatrix} 4\\3 \end{bmatrix}$$

$$\overrightarrow{m} = \begin{bmatrix} 5\\5 \end{bmatrix}$$

Selanjutnya dapat kita masukan ke dalam rumus *Cosine Similarity*
* $$b \cdot c$$ = (4 x 5) + (3 x 5) = 35
* ||b|| = $$\sqrt{4^{2} + 3^{2}}$$ = 5
* ||c|| = $$\sqrt{5^{2} + 5^{2}}$$ = $$5\sqrt{2}$$
* cosine similarity = $$\tfrac{35}{5\times 5\sqrt{2}}\approx0.989$$

##### Kelebihan
* Tidak memerlukan data apapun terhadap pengguna
* Dapat merekomendasikan item khusus

##### Kelemahan
* Membutuhkan banyak pengetahuan suatu domain
* Membuat rekomendasi berdasarkan minat pengguna yang ada saja

##### Recommendation Result
Berdasarkan deskripsi dari film "*The Godfather*" maka didapatkan rekomendasi film yang memiliki deskripsi yang mirip sebagai berikut :

| title                   | vote_average | similarity_score |
|-------------------------| ---------- | ------------------|
| The Godfather: Part II  |8.3 | 0.220060 |
| The Family              |6.1 | 0.100294 |
| Made                    |6.3 | 0.067618 |
| Johnny Dangerously      |6.3 | 0.065622 |
| Shanghai Triad          |6.5 | 0.056142 |
| Fury                    |7.5 | 0.056028 |
 | American Movie          |7.7 | 0.055023 |
| The Godfather: Part III | 7.1| 0.050235 |
| 8 Women                 |6.9 | 0.047508 |
| Summer of Sam           |6.3 | 0.045952 |
Tabel 2, hasil rekomendasi dari pendekatan *Content Based Filtering*

Dapat kita lihat bahwa ada beberapa film yang mungkin memiliki deskripsi yang tidak mirip dengan *The Godfather* muncul dalam rekomendasi. Inilah salah satu kelemahan dari sistem rekomendasi dengan pendekatan content based filtering karena tidak memberikan rekomendasi sesuai dengan preferensi dari pengguna. Diharapkan pendekatan ini dapat meningkatkan *experience user* dalam mencari film yang mirip.

### *User Collaborative Filtering*
*User Collaborative Filtering* adalah suatu metode dalam membuat rekomendasi otomatis untuk memprediksi ketertarikan atau selera seseorang terhadap suatu *item* dengan cara mengumpulkan informasi dari *user* terkait. Sebagai contoh jika *user A* suka dengan genre *drama, comedy* dan *user B* suka dengan genre *comedy*, maka kemungkinan besar *user B* akan direkomendasikan film dengan genre drama juga.

Dalam pembuatannya, *User Collaborative Filtering* menggunakan library khusus untuk sistem rekomendasi dari [*Surprise*](https://surpriselib.com/) dimana library yang akan digunakan adalah SVD (Singular Value Decomposition). SVD adalah teknik faktorisasi matriks, dimana SVD bekerja untuk mengurangi jumlah fitur dari kumpulan data dengan cara mengurangi dimensi ruang dari dimensi-N ke dimensi-K (dimana K < N)[5]. Hal ini menggunakan struktur matriks dimana setiap baris mewakili *user*, dan setiap kolom mewakili *item*. Elemen matriks ini adalah rating yang diberikan kepada *item* oleh *user*. Rumus matematis SVD adalah sebagai berikut :

$$A = USV^{T}$$

dimana :
* A : m x n matriks
* U : m x r matriks singular kiri ortogonal, yang merepresentasikan hubungan antara hubungan dengan faktor laten
* S : r x r matriks diagonal, yang menggambarkan kekuatan setiap faktor laten
* V : r x n matriks singular kanan diagonal, yang menunjukan kesamaan antara *item* dan faktor laten

Faktor laten di sini adalah karateristik *item*, dalam kasus kita adalah genre film. SVD mengurangi dimensi matriks A dengan mengekstrak faktor latennya. Yang memetakan setiap pengguna dan setiap *item* ke dalam ruang laten r-dimensi. Pemetaan ini memfasilitasi representasi yang jelas dari hubungan antara *item* dan *user*.

##### Kelebihan
* Menghasilkan rekomendasi yang berkualitas baik karena hasil rekomendasi berdasarkan preferensi *user*

##### Kelemahan
* Kompleksitas perhitungan akan semakin bertambah seiring dengan bertambahnya pengguna sistem, semakin banyak pengguna (user) yang menggunakan sistem maka proses perekomendasian akan semakin lama 

##### Recommendation Result
Berdasarkan *userId* 1 yang ingin mencari rekomendasi film mirip *Avatar*, maka didapatkan hasil rekomendasi sebagai berikut :
| title                           | vote_count | vote_average | year |
|---------------------------------|------------|--------------|------|
| A Grand Day Out                 | 199.0      | 7.4          | 1990  |
| The Matrix                      | 9079.0     | 7.9          | 1999 |
 | A Trip to the Moon              | 314.0      | 7.9          | 1902 |
 | Green Zone                      | 730.0      | 6.4          | 2010 |
 | A Simple Plan                   | 191.0      | 6.9          | 1998 |
 | The Hidden                      | 85.0       | 6.7          | 1987 |
 | Hanna                           | 1284.0     | 6.5          | 2011 |
 | Ambush                          | 13.0       | 6.3          | 1999 |
| Pandora and the Flying Dutchman | 19.0       | 6.5          | 1951 |
| Heaven Knows, Mr. Allison       | 27.0       | 6.8          | 1957 |
Tabel 3, hasil rekomendasi dari pendekatan *Collaborative User Filtering*

Dapat kita lihat bahwa ada beberapa film yang tahun perilisannya sangat lama sekali yaitu 1902. Hal tersebut bisa terjadi jika film tersebut sesuai dengan preferensi dari *user*. Sehingga diharapkan pendekatan ini dapat meningkatkan *user experience* dalam menentukan preferensi film *user* sehingga *user* tidak perlu mencari film lagi yang sesuai dengan preferensi mereka.

Sehingga kesimpulan dari proyek ini adalah perekomendasian film kepada *user* ditentukan oleh kesamaan film satu dengan yang lain, dalam kasus ini adalah genre dari film. Namun preferensi *user* juga dapat menjadi pertimbangan dalam pemberian rekomendasi film, dilihat dari rating *user* terhadap sebuah film atau genre.

# Evaluation

Untuk metric evaluasi proyek ini akan menggunakan 2 metric, yaitu *precision@k* untuk *Content Based Filtering* dan *RMSE* untuk *User Collaborative Filtering*.

### *Content Based Filtering*
Untuk evaluasi dari sistem rekomendasi dengan pendekatan content based filtering kita dapat menggunakan salah satu metric yaitu precision@K. Apa itu precision? Precision adalah perbandingan antara True Positive (TP) dengan banyaknya data yang diprediksi positif. Atau juga bisa ditulis secara matematis sebagai berikut :

*precision* = $$\tfrac{TP}{TP + FP}$$

dimana : 
* TP = *true positive* 
* FP = *false positive*

Namun pada sistem rekomendasi kita tidak akan menggunakan True positive atau False Positive melainkan rating yang diberikan pada buku untuk menentukan buku yang direkomendasikan relevan atau tidak. Dengan rumus sebagai berikut :

*precision@k* = $$\tfrac{totalfrecommended item that relevant}{of recommended item}$$

Lalu apa definisi relevan? Bisa kita definisikan sebagai berikut :
* Relevan: Rating > 5
* Tidak relevan: Rating < 5

Angka 5 tersebut merupakan nilai arbiter dimana kita mengatakan bahwa nilai yang berada diatas akan dianggap relevan. Ada banyak metode untuk memilih nilai itu, tetapi untuk proek ini, kita akan menggunakan angka 5 sebagai ambang batas, yang berarti setiap buku yang memiliki rating di atas 5, akan dianggap relevan dan di bawahnya tidak akan relevan.

* Pertama kita memilih nilai K yaitu 10 (sesuai total hasil rekomendasi)
* Lalu kita akan menentukan relevansi threshold yaitu 5
* Kemudian kita akan memfilter semua buku rekomendasi sesuai threshold
* Terakhir hitung dengan rumus precision@K di atas

Dari hasil rekomendasi yang dihasilkan, didapatkan precision sebagai berikut :

precision@10 = $$\tfrac{10}{10}$$ = 1

Karena seluruh film yang direkomendasikan memiliki rating diatas 5, maka *precision* dari model *Content Based Filtering* adalah 100%

### *Collaborative User Filtering*
Karena pendekatan ini kita menggunakan libary dari [*Surprise*](https://surpriselib.com/) yaitu SVD (Singular Value Decomposition), maka untuk metric kita dapat gunakan *RMSE* dan *MAE* yaitu untuk meminimalisir *error* dari rekomendasi. Dengan menggunaakan *cross validation* didapatkan *error* sebagai berikut :

|           | Fold 1              | Fold 2              | Fold 3              | Fold 4             | Fold 5              |
|-----------|---------------------|---------------------|---------------------|--------------------|---------------------|
| test_rmse | 0.89921598          | 0.89444179          | 0.89156229          | 0.90112036         | 0.89424624          |
| test_mae  | 0.69409506          | 0.69063001          | 0.68839102          | 0.69220899         | 0.68802139          |
| fit_time  | 1.0370290279388428  | 1.106001853942871   | 1.063995361328125   | 1.0370023250579834 | 0.99300217628479    |
| test_time | 0.12396955490112305 | 0.12399888038635254 | 0.12999677658081055 | 0.1179966926574707 | 0.11699914932250977 |
Tabel 4, hasil *cross validation* dari pendekatan *Collaborative User Filtering*

Sistem rekomendasi dapat memberikan rekomendasi terhadap *user* aktif dengan cara menghitung rating dengan kondisi beberapa *user* lainnya yang pernah memberi rating pada film yang akan diprediksi *user* aktif. Dari evaluasi yang telah dilakukan dengan metode *precision@k* (*Content Based Filtering*) didapatkan hasil yang sangat memuaskan yaitu 1, sedangkan untuk *User Collaborative Filtering* dapat dilakukan dengan metode *RMSE* dan *MAE* dengan menggunakan *cross validation* sebanyak 5 *fold* dengan perhitungan 0.90112036 dan 0.69220899 secara berurutan.

# Referensi (APA7)  :
---

1. Fajriansyah, M., Adikara, P. P., &amp; Widodo, A. W. (2021). Sistem Rekomendasi Film Menggunakan Content Based Filtering, 5(6), 2188–2199. 
2. Wijaya, A. E., &amp; Alfian, D. (2018). SISTEM REKOMENDASI LAPTOP MENGGUNAKAN COLLABORATIVE FILTERING DAN CONTENT-BASED FILTERING, 12(1), 11–27.
3. Agustian, E. R., Munir, &amp; Nugroho, E. P. (2020). Sistem Rekomendasi Film Menggunakan Metode Collaborative Filtering Dan K-Nearest  Neighbors, 3(1), 18–21. 
4. Using cosine similarity to build a movie recommendation system. (n.d.). Retrieved November 7, 2022, from https://towardsdatascience.com/using-cosine-similarity-to-build-a-movie-recommendation-system-ae7f20842599
5. Kumar, D. V. (2022, March 26). Singular value decomposition (SVD) &amp; its application in Recommender System. Analytics India Magazine. Retrieved November 7, 2022, from https://analyticsindiamag.com/singular-value-decomposition-svd-application-recommender-system/ 