import os
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from keras.src.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# Definir los directorios
dataset_path = '"C:/Users/Dragu/Desktop/archive.zip"'
cropped_path = "C:/Users/Dragu/PycharmProjects/RetoComputerVision/archive/cropped"
simplified_path = "C:/Users/Dragu/PycharmProjects/RetoComputerVision/archive/simplified"

IMG_SIZE = (128, 128)  # Tamaño estándar para redes neuronales

# Cargar imágenes y sus etiquetas
X, Y = [], []

# Cargar imágenes
for img_id in os.listdir(cropped_path)[:1000]:
    cropped_img = cv2.imread(os.path.join(cropped_path, img_id))
    simplified_img = cv2.imread(os.path.join(simplified_path, img_id))

    if cropped_img is None or simplified_img is None:
        print(f" Problema al cargar {img_id}")
    else:
        # Redimensionar las imágenes
        cropped_img = cv2.resize(cropped_img, IMG_SIZE)
        simplified_img = cv2.resize(simplified_img, IMG_SIZE)

        # Promediar las imágenes
        final_img = (cropped_img + simplified_img) / 2

        X.append(final_img)

        # Obtener etiqueta a partir del nombre del archivo (aquí se asigna una clase numérica)
        label = int(img_id.split('.')[0]) % 20  # Ejemplo simple para 20 clases (mod 20)
        Y.append(label)

X = np.array(X)
Y = np.array(Y)

print(f"Tamaño de X: {X.shape}")
print(f"Tamaño de Y: {Y.shape}")


Y = to_categorical(Y, num_classes=20)

# Dividir el dataset en entrenamiento y validación
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalizar las imágenes
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

print(f"Tamaño de X_train: {X_train.shape}")
print(f"Tamaño de Y_train: {Y_train.shape}")
print(f"Tamaño de X_val: {X_val.shape}")
print(f"Tamaño de Y_val: {Y_val.shape}")

# Creación de generadores de datos para aumento de datos
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Ajustar el generador de datos en las imágenes de entrenamiento
datagen.fit(X_train)

# Usar ResNet50 como base para transfer learning
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Crear el modelo
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(20, activation='softmax')  # 20 clases
])

# Compilar el modelo con un optimizador con tasa de aprendizaje ajustada
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con aumento de datos
history = model.fit(datagen.flow(X_train, Y_train, batch_size=32),
                    validation_data=(X_val, Y_val),
                    epochs=10)

# Graficar la precisión y la pérdida durante el entrenamiento
plt.figure(figsize=(12, 6))

# Precisión de entrenamiento y validación
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Epochs')
plt.ylabel('Precisión')
plt.legend()

# Pérdida de entrenamiento y validación
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Epochs')
plt.ylabel('Pérdida')
plt.legend()

plt.show()

# Evaluar el modelo
train_accuracy = model.evaluate(X_train, Y_train, verbose=0)
val_accuracy = model.evaluate(X_val, Y_val, verbose=0)

print(f"Exactitud de entrenamiento: {train_accuracy[1]}")
print(f"Exactitud de validación: {val_accuracy[1]}")



