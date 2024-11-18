
# Bibliotecas necesarias
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from scipy.sparse import csr_matrix
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import KMeansSMOTE
import nltk
import contractions
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


#!pip install contractions

# Configurar palabras vacías (stopwords)
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}
lemmatizer = WordNetLemmatizer()

# Preprocesamiento de texto
def preprocess_text(text):
    """Preprocesar texto para el análisis de sentimientos"""
    text = contractions.fix(text)  # Expandir contracciones
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Eliminar URLs
    text = re.sub(r'<.*?>', '', text)  # Eliminar HTML
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuación
    text = re.sub(r'\d+', '', text)  # Eliminar números
    tokens = word_tokenize(text.lower())  # Convertir a minúsculas y tokenizar
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_dataset():
    """Cargar el dataset de emociones"""
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            global data
            data = pd.read_csv(file_path)
            messagebox.showinfo("Carga exitosa", f"Archivo cargado: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Hubo un problema al cargar el archivo.\n{e}")

    return data

# Definir funciones de activación y pérdida
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def categorical_cross_entropy(y, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y * np.log(y_pred), axis=1))

# Entrenamiento con mini-batch gradient descent
def train_gradient_descen(X, y, learning_rate, epochs, batch_size=128, tol=1e-6, optimizer="adam"):
    weights = np.random.randn(X.shape[1], y.shape[1]) * 0.01  # Pesos inicializados aleatoriamente
    bias = np.zeros((y.shape[1],))  # Sesgo inicial
    losses = []

    # Inicialización de Adam
    m_w, v_w = np.zeros_like(weights), np.zeros_like(weights)
    m_b, v_b = np.zeros_like(bias), np.zeros_like(bias)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(epochs):
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Predicción
            linear_model = X_batch.dot(weights) + bias
            y_pred = softmax(linear_model)

            # Calcular gradientes
            error = y_pred - y_batch
            gradients_w = X_batch.T.dot(error) / batch_size
            gradients_b = np.mean(error, axis=0)

            # Optimización con Adam
            if optimizer == "adam":
                t += 1
                m_w = beta1 * m_w + (1 - beta1) * gradients_w
                v_w = beta2 * v_w + (1 - beta2) * (gradients_w ** 2)
                m_b = beta1 * m_b + (1 - beta1) * gradients_b
                v_b = beta2 * v_b + (1 - beta2) * (gradients_b ** 2)

                m_w_hat = m_w / (1 - beta1 ** t)
                v_w_hat = v_w / (1 - beta2 ** t)
                m_b_hat = m_b / (1 - beta1 ** t)
                v_b_hat = v_b / (1 - beta2 ** t)

                weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
                bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)
            else:  # SGD
                weights -= learning_rate * gradients_w
                bias -= learning_rate * gradients_b

        # Calcular pérdida
        loss = categorical_cross_entropy(y, softmax(X.dot(weights) + bias))
        losses.append(loss)

        # Early Stopping
        if epoch > 1 and abs(losses[-1] - losses[-2]) < tol:
            print(f"Entrenamiento detenido en la época {epoch}, pérdida: {loss}")
            break

        if epoch % 10 == 0:
            print(f"Época {epoch}, Pérdida: {loss}")

    return weights, bias, losses



# Predicción
def predict(X, weights, bias):
    linear_model = X.dot(weights) + bias
    y_pred = softmax(linear_model)
    return np.argmax(y_pred, axis=1)

def train_model():
    """Entrenar el modelo"""
    try:
        if data is None:
            raise Exception("No se ha cargado un dataset")
        
        # Preprocesar datos
        data['clean_text'] = data['text'].apply(preprocess_text)
        X = data['clean_text']
        y = data['label']
        
        # Mostrar resultados preprocesados
        print(data.head())
        
        # Dividir datos
        global X_test_vec, y_test, weights, bias, vectorizer
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Vectorización
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # One-hot encoding
        num_classes = len(np.unique(y_train))
        y_train_one = np.eye(num_classes)[y_train]
        
        # Entrenar modelo
        weights, bias, _ = train_gradient_descen(X_train_vec, y_train_one, learning_rate=0.01, epochs=100, batch_size=128)
        messagebox.showinfo("Éxito", "Entrenamiento completado.")
        
    except Exception as e:
        messagebox.showerror("Error", f"Hubo un problema al entrenar el modelo.\n{e}")
        
def evaluate_model():
    """Evaluar el modelo"""
    try:
        y_pred = predict(X_test_vec, weights, bias)
        accuracy = accuracy_score(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        class_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de Confusión")
        plt.show()
        
        messagebox.showinfo("Informe de Clasificación", f"Precisión: {accuracy}\n\n{cr}")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo evaluar el modelo.\n{e}")

def predict_emotion():
    """Predecir la emoción de una frase ingresada"""
    try:
        sentence = entry_sentence.get()
        if not sentence:
            raise ValueError("Por favor ingresa una frase.")
        
        processed_text = preprocess_text(sentence)
        test_vector = vectorizer.transform([processed_text])
        predicted_class_index = predict(test_vector, weights, bias)
        class_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        predicted_emotion = class_labels[predicted_class_index[0]]
        
        messagebox.showinfo("Predicción", f"Emoción predicha: {predicted_emotion}")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo realizar la predicción.\n{e}")

# Interfaz gráfica
root = tk.Tk()
root.title("Análisis de Emociones")
root.geometry("400x400")

# Botones
btn_load = tk.Button(root, text="Cargar archivo CSV", command=load_dataset)
btn_load.pack(pady=10)

btn_train = tk.Button(root, text="Entrenar modelo", command=train_model)
btn_train.pack(pady=10)

btn_evaluate = tk.Button(root, text="Evaluar modelo", command=evaluate_model)
btn_evaluate.pack(pady=10)

lbl_sentence = tk.Label(root, text="Ingresa una frase para predecir la emoción:")
lbl_sentence.pack(pady=10)

entry_sentence = tk.Entry(root, width=50)
entry_sentence.pack(pady=10)

btn_predict = tk.Button(root, text="Predecir emoción", command=predict_emotion)
btn_predict.pack(pady=10)

# Variables globales
data = None
X_test_vec = None
y_test = None
weights = None
bias = None
vectorizer = None

root.mainloop()

"""
# Análisis inicial
print(data.head())  # Vista rápida de los datos
print(data['label'].value_counts())  # Distribución de etiquetas

print(f'The Shape Of Data Is : {data.shape}') # Forma de los datos
data.isnull().sum()  # Verificar valores nulos

data.duplicated().sum()  # Verificar valores duplicados


# Renombrar columnas
data.rename(columns={"text": "Texto", "label": "Emocion"}, inplace=True)

# Borrar columna de index si existe
if 'Unnamed: 0' in data.columns:
    data.drop('Unnamed: 0', axis=1, inplace=True)

data.head()  # Vista rápida de los datos

data_aux = data.copy()  # Copia de seguridad

# Renombrar las emociones
data_aux['Emocion'] = data_aux['Emocion'].replace({0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'})



# Analisis Grafico
count = data_aux['Emocion'].value_counts()

# Crear columna con dos subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')

# Grafico de barras
palette = ["#FF0000", "#FFA500", "#FFFF00", "#008000", "#0000FF", "#4B0082"]
sns.set_palette(palette)
ax[0].pie(count, labels=count.index, autopct='%1.1f%%', startangle=90, shadow=True)
ax[0].set_title('Distribución de Emociones')

sns.barplot(x=count.index, y=count.values, ax=ax[1], palette=palette)
ax[1].set_title('Conteo de emociones')

plt.tight_layout()
plt.show()


# Separa el data set para visualizar las emociones
sadness = data_aux[data_aux['Emocion'] == 'sadness']
joy = data_aux[data_aux['Emocion'] == 'joy']
love = data_aux[data_aux['Emocion'] == 'love']
anger = data_aux[data_aux['Emocion'] == 'anger']
fear = data_aux[data_aux['Emocion'] == 'fear']
surprise = data_aux[data_aux['Emocion'] == 'surprise']


# Crear wordclouds
def wordcloud(data, title):
    wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(' '.join(data['Texto']))
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()

wordcloud(sadness, 'Sadness')
wordcloud(joy, 'Joy')
wordcloud(love, 'Love')
wordcloud(anger, 'Anger')
wordcloud(fear, 'Fear')
wordcloud(surprise, 'Surprise')

"""


