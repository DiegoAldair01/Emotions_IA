import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# Word Cloud
from wordcloud import WordCloud
# from textacy import preprocessing
from nltk.stem.snowball import SnowballStemmer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Cargar el dataset
data = pd.read_csv("emotions_dataset.csv")

# Análisis inicial
print(data.head())  # Vista rápida de los datos
print(data['label'].value_counts())  # Distribución de etiquetas

print(f'The Shape Of Data Is : {data.shape}') # Forma de los datos

data.isnull().sum()  # Verificar valores nulos

data.duplicated().sum()  # Verificar valores duplicados

# Eliminar valores duplicados
data.drop_duplicates()

# Renombrar columnas
data = data.rename(columns={"text": "Texto", "label": "Emocion"})
# Borrando columna de index
data = data.drop('Unnamed: 0', axis=1)



data.head() # Vista rápida de los datos

data_aux = data.copy()  # Copia de seguridad

# Renombrar las emociones {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
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



# Preprocesamiento de Texto

# Convertir texto a minus
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Eliminar URLs
    text = re.sub(r'<.*?>', '', text) # Eliminar etiquetas HTML
    text = re.sub(r'[^\w\s]', '', text) # Eliminar puntuación
    text = re.sub(r'\d+', '', text)  # Remover números
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Eliminar caracteres especiales
    text = re.sub(r'\s+', ' ', text) # Eliminar espacios en blanco
    tokens = word_tokenize(text) # Tokenizar texto
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] # Lematizar palabras
    return ' '.join(tokens)

data['clean_text'] = data['Texto'].apply(preprocess_text)
print(data.head())





# División del dataset
X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], data['label'], test_size=0.2, random_state=42)

# Convertir textos a vectores
vectorizer = TfidfVectorizer(max_features=5000) # Seleccionar las 5000 palabras más frecuentes
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)