# LAB_IS_Final


Repo ini dibuat berdasarkan penugasan `final project internship lab Intelligent System`.
Project ini bertujuan untuk membuat sebuah job recommender system dengan menggabungkan hasil analisis cv dengan job yang sedang trending.

Dataset:
1. Dataset teks CV untuk training didapat dari huggingface: https://huggingface.co/datasets/ahmedheakl/resume-atlas
2. Dataset Scrapping, Hasil scrapping Linkedin, berisikan ﻿job_id,Title,Company,Location,Date,Category
3. CV user sebagai data testing end user

Penjelasan File:
1. cv_analyst.ipynb - file utama dalam pembuatan model klasifikasi cv
2. ocr_for_cv.py - file yang berisikan kode untuk melakukan ocr terhadap data testing berupa cv image
3. scrapping_job_list.ipynb - file yang berisikan kode untuk melakukan scrapping
4. preprocessing_pip.py - file yang berisikan pipeline untuk memproses serta melakukan cleaning data input user yang telah di ekstraksi melalui ocr
5. predict_streamlit - file yang berisikan kode penggabungan bobot serta streamlit untuk end user sederhana

Evaluasi Model CV:
| Nama Model | | Precision (%) | Recall (%) | F1-Score (%) |
|---|---|---|---|---|
| **DistilBERT** | Macro Average | 89.16 | 89.42 | 89.14 |
| | Weighted Average | 90.17 | 89.89 | 89.92 |
| | Accuracy | | 89.89 | |
| **TF-IDF + Random Forest (n_est = 100)** | Macro Average | 78.4 | 78 | 77.5 |
| | Weighted Average | 78.3 | 78.2 | 77.6 |
| | Accuracy | | 78.2 | |
| **TF-IDF + Random Forest (n_est = 300)** | Macro Average | 78.6 | 78.2 | 77.7 |
| | Weighted Average | 78.5 | 78.5 | 77.8 |
| | Accuracy | | 78.5 | |


