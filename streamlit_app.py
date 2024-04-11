import streamlit as st
from joblib import load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
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
    # Токенизация
    tokens = word_tokenize(text)
    # Удаление стоп-слов
    stop_words = set(stopwords.words('russian'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # Склеивание токенов обратно в текст
    preprocessed_text = ' '.join(lemmatized_tokens)
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
