# Bibliotecas necesarias
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from scipy.sparse import csr_matrix, lil_matrix
from wordcloud import WordCloud
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sympy.logic.boolalg import And, Not, Or
import nltk
import contractions
import re
import networkx as nx
import threading

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Configurar palabras vacías (stopwords)
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}
lemmatizer = WordNetLemmatizer()

# Variables globales
data = None
X_test_vec = None
y_test = None
weights = None
bias = None
vectorizer = None
semantic_network = None

# Preprocesamiento de texto
def preprocess_text(text):
    """Preprocesar texto para el análisis"""
    text = contractions.fix(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Enriquecer texto con palabras conectadas en la red semántica
    if semantic_network:
        enriched_tokens = []
        for token in tokens:
            enriched_tokens.append(token)
            if token in semantic_network.nodes:
                enriched_tokens.extend(list(semantic_network.neighbors(token)))
        tokens = enriched_tokens

    return ' '.join(tokens)

# Construir la red semántica
def build_semantic_network(emotions):
    """Crear una red semántica para las emociones"""
    G = nx.Graph()
    for emotion in emotions:
        G.add_node(emotion)
        for syn in wn.synsets(emotion):
            for lemma in syn.lemmas():
                word = lemma.name()
                if word != emotion:
                    G.add_edge(emotion, word)
    return G

def enrich_with_semantic_network_optimized(X, vectorizer, semantic_network):
    """Enriquecer los datos TF-IDF con la red semántica, optimizado para palabras relevantes."""
    # Crear una copia de la matriz original
    X_enriched = X.copy()

    # Obtener el vocabulario del vectorizador
    vocab = vectorizer.vocabulary_

    # Iterar solo sobre palabras que aparecen en los datos
    for word in semantic_network.nodes:
        if word in vocab:
            word_idx = vocab[word]
            neighbors = semantic_network.neighbors(word)

            # Iterar sobre los vecinos y enriquecer las columnas relevantes
            for neighbor in neighbors:
                if neighbor in vocab:
                    neighbor_idx = vocab[neighbor]
                    # Sumar las frecuencias de las palabras relacionadas
                    X_enriched[:, neighbor_idx] += X[:, word_idx]

    return X_enriched


# Cargar dataset
def load_dataset():
    """Cargar el archivo CSV"""
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            global data
            data = pd.read_csv(file_path)
            messagebox.showinfo("Carga exitosa", f"Archivo cargado: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Hubo un problema al cargar el archivo.\n{e}")

# Definir funciones de activación y pérdida
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def categorical_cross_entropy(y, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y * np.log(y_pred), axis=1))

# Entrenamiento con red semántica
from scipy.sparse import lil_matrix

def train_gradient_descen(X, y, learning_rate, epochs, batch_size=128, tol=1e-6, optimizer="adam"):
    """Entrenamiento con mini-batch gradient descent."""
    weights = np.random.randn(X.shape[1], y.shape[1]) * 0.01  # Inicialización de pesos
    bias = np.zeros((y.shape[1],))  # Inicialización de sesgo
    losses = []

    # Inicialización de Adam
    m_w, v_w = np.zeros_like(weights), np.zeros_like(weights)
    m_b, v_b = np.zeros_like(bias), np.zeros_like(bias)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    # Entrenamiento por épocas
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

            # Actualización de parámetros con Adam
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

        if epoch > 1 and abs(losses[-1] - losses[-2]) < tol:
            print(f"Entrenamiento detenido en la época {epoch}, pérdida: {loss}")
            break

        if epoch % 10 == 0:
            print(f"Época {epoch}, Pérdida: {loss}")

    return weights, bias, losses




# Función para entrenar el modelo
def train_model_thread():
    try:
        global X_test_vec, y_test, weights, bias, vectorizer, semantic_network
        
        #Mostrar mensaje de entrenamiento en proceso
        status_label.config(text="Entrenamiento en proceso... (Esto puede tardar algunos minutos)")
        root.update_idletasks()
        
        if data is None:
            raise Exception("No se ha cargado un dataset")

        # Construir red semántica
        emotions = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
        semantic_network = build_semantic_network(emotions)
        print("Red semántica construida.")
        
        # Preprocesar texto
        data['clean_text'] = data['text'].apply(preprocess_text)
        X = data['clean_text']
        y = data['label']
        print("Texto preprocesado.")  
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print("Datos divididos.")
        
        # Vectorización
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        print("Datos vectorizados.")
        
        # Enriquecer los datos con la red semántica (optimizando para palabras relevantes)
        X_train_vec = enrich_with_semantic_network_optimized(X_train_vec, vectorizer, semantic_network)
        X_test_vec = enrich_with_semantic_network_optimized(X_test_vec, vectorizer, semantic_network)
        print("Datos enriquecidos con red semántica.")
        
        # One-hot encoding
        num_classes = len(np.unique(y_train))
        y_train_one = np.eye(num_classes)[y_train]
        print("One-hot encoding completado.")
        
        # Entrenar modelo
        weights, bias, _ = train_gradient_descen(
            X_train_vec, y_train_one, learning_rate=0.01, epochs=100, batch_size=128
        )
        status_label.config(text="Entrenamiento completado.")
        messagebox.showinfo("Éxito", "Entrenamiento completado.")
    except Exception as e:
        status_label.config(text="Error en el entrenamiento.")
        messagebox.showerror("Error", f"Hubo un problema al entrenar el modelo.\n{e}")

def train_model():
    """Función para iniciar el entrenamiento en un hilo separado."""
    thread = threading.Thread(target=train_model_thread)
    thread.start()

        
def evaluate_model():
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

# Predicción
def predict(X, weights, bias):
    linear_model = X.dot(weights) + bias
    y_pred = softmax(linear_model)
    return np.argmax(y_pred, axis=1)

def predict_emotion():
    try:
        # Obtener la frase de entrada
        sentence = entry_sentence.get()
        if not sentence:
            raise ValueError("Por favor ingresa una frase.")

        # Preprocesar la frase
        processed_text = preprocess_text(sentence)

        # Vectorizar la frase
        test_vector = vectorizer.transform([processed_text])

        # Enriquecer el vector con la red semántica
        test_vector = enrich_with_semantic_network_optimized(test_vector, vectorizer, semantic_network)

        # Realizar la predicción
        predicted_class_index = predict(test_vector, weights, bias)
        class_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        predicted_emotion = class_labels[predicted_class_index[0]]

        # Mostrar la predicción
        messagebox.showinfo("Predicción", f"Emoción predicha: {predicted_emotion}")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo realizar la predicción.\n{e}")


# Interfaz gráfica
def create_interface():
    global status_label, entry_sentence, root
    root = tk.Tk()
    root.title("Análisis de Emociones")
    root.geometry("500x600")
    root.configure(bg="#e8f4fc")  # Fondo azul claro

    # Título principal
    title_label = tk.Label(
        root,
        text="Análisis de Emociones",
        font=("Helvetica", 20, "bold"),
        bg="#e8f4fc",
        fg="#004c99"
    )
    title_label.pack(pady=20)

    # Frame para botones
    frame_buttons = tk.Frame(root, bg="#e8f4fc")
    frame_buttons.pack(pady=20)

    button_style = {
        "font": ("Helvetica", 12),
        "bg": "#007ACC",
        "fg": "white",
        "relief": "flat",
        "width": 20,
        "bd": 4
    }

    # Botón para cargar dataset
    btn_load = tk.Button(frame_buttons, text="📂 Cargar archivo CSV", command=load_dataset, **button_style)
    btn_load.pack(pady=10)

    # Botón para entrenar modelo
    btn_train = tk.Button(frame_buttons, text="🚀 Entrenar modelo", command=train_model, **button_style)
    btn_train.pack(pady=10)

    # Botón para evaluar el modelo
    btn_evaluate = tk.Button(frame_buttons, text="📊 Evaluar modelo", command=evaluate_model, **button_style)
    btn_evaluate.pack(pady=10)

    # Label para mostrar el estado del entrenamiento
    status_label = tk.Label(
        root,
        text="",  # Mensaje inicial vacío
        font=("Helvetica", 12),
        bg="#e8f4fc",
        fg="#004c99"
    )
    status_label.pack(pady=10)

    # Frame para la predicción
    frame_predict = tk.Frame(root, bg="#e8f4fc")
    frame_predict.pack(pady=20)

    lbl_sentence = tk.Label(
        frame_predict,
        text="💬 Ingresa una frase para predecir la emoción:",
        font=("Helvetica", 12),
        bg="#e8f4fc",
        fg="#004c99"
    )
    lbl_sentence.pack(pady=5)

    # Campo de texto para ingresar la frase
    entry_sentence = ttk.Entry(frame_predict, width=40, font=("Helvetica", 12))
    entry_sentence.pack(pady=5)

    # Botón para predecir la emoción
    btn_predict = tk.Button(root, text="🔍 Predecir emoción", command=predict_emotion, **button_style)
    btn_predict.pack(pady=20)

    # Footer
    footer_label = tk.Label(
        root,
        text="© 2024 Gerardo Arredondo, Daniela Castro, Luis Cruz, Diego García",
        font=("Helvetica", 10, "italic"),
        bg="#e8f4fc",
        fg="#666"
    )
    footer_label.pack(pady=10)

    root.mainloop()


# Crear la interfaz
create_interface()
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


