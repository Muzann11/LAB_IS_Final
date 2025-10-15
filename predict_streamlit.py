import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import pandas as pd
from preprocessing_pip import CVPreprocessor 
from ocr_for_cv import ocr_image

@st.cache_resource
def load_resources(model_path='./model'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = CVPreprocessor(tokenizer_path='./tokenizer', label_encoder_path='label_encoder.pkl')
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return model, preprocessor, device

trend_df = pd.read_csv("job_trend_scaled.csv")
trend_weights = dict(zip(trend_df['category'], trend_df['scaled Count']))

st.title("Klasifikasi CV dengan Late Fusion")
st.subheader("Input CV")

uploaded_file = st.file_uploader("Unggah Gambar CV (JPG/PNG)", type=["jpg", "jpeg", "png"])
ocr_text = ""
if uploaded_file is not None:
    with st.spinner("Membaca teks dari gambar..."):
        ocr_text = ocr_image(uploaded_file)
        if not ocr_text.strip():
             st.warning("OCR berhasil dieksekusi, tetapi tidak menemukan teks")
             
model, preprocessor, device = load_resources()

if st.button("Prediksi Kategori CV"):
    if not ocr_text.strip():
        st.warning("Tolong masukkan teks CV terlebih dahulu.")
    else:
        with st.spinner("Memproses dan memprediksi..."):
            tokens = preprocessor.preprocess_for_prediction(ocr_text)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top10_prob, top10_indices = torch.topk(probabilities, 10)

            top10_prob = top10_prob.cpu().squeeze().tolist()
            top10_indices = top10_indices.cpu().squeeze().tolist()

            st.subheader("Probabilitas Awal (Tanpa Pembobotan)")
            df_raw = pd.DataFrame({
                "Rank": range(1, 11),
                "Kategori": [preprocessor.decode_prediction(i) for i in top10_indices],
                "Prob (%)": [p * 100 for p in top10_prob]
            })
            st.table(df_raw)

            # implementasi late fusion
            bobot_df = pd.read_csv("job_trend_scaled.csv")
            df_merged = df_raw.merge(bobot_df, left_on="Kategori", right_on="category", how="left")
            a = 0.3
            df_merged["Adjusted Prob (%)"] = a * df_merged["scaled Count"] + (1 - a) * (df_merged["Prob (%)"] / 100)
            df_merged["Adjusted Prob (%)"] *= 100

            st.subheader("Setelah Diterapkan Bobot (Weighted Late Fusion)")
            top10 = df_merged.sort_values("Adjusted Prob (%)", ascending=False).head(10)
            st.table(top10[["Rank", "Kategori", "Prob (%)", "scaled Count", "Adjusted Prob (%)"]])

            st.success(f"Kategori Teratas: **{top10.iloc[0]['Kategori']}** ({top10.iloc[0]['Adjusted Prob (%)']:.2f}%)")    