import streamlit as st
from joblib import load
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузка модели из файла
model = load('sentiment_classifier.joblib')

# Загрузка tfidf_vectorizer из файла
tfidf_vectorizer = load('tfidf_vectorizer.joblib')

# Заголовок приложения
st.title('Тональность текста')

# Текстовое поле для ввода данных
text_input = st.text_area("Введите текст", "")

def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление пунктуации
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Простая токенизация с использованием регулярного выражения
    tokens = re.findall(r'\b\w+\b', text)
    # Склеивание токенов обратно в текст
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Кнопка для запуска предсказания
if st.button("Предсказать"):
    # Предобработка введенного пользователем текста
    preprocessed_text = preprocess_text(text_input)
    # Преобразование текста в вектор признаков
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    # Предсказание тональности текста
    prediction = model.predict(text_tfidf)[0]
    # Отображение предсказания
    st.write(f"Тональность текста: {prediction}")
