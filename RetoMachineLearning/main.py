# Importar las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


# Cargar el dataset
df = pd.read_csv('C:\\Users\\Dragu\\Desktop\\heart_attack_risk_dataset.csv')


# Mostrar información del dataset
print(df.info())
print(df.head())

# Convertir variables categóricas a numéricas
encoder = LabelEncoder()
categorical_cols = ['Gender', 'Physical_Activity_Level', 'Stress_Level',
                    'Chest_Pain_Type', 'Thalassemia', 'ECG_Results']

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Convertir la columna objetivo
df['Heart_Attack_Risk'] = encoder.fit_transform(df['Heart_Attack_Risk'])

# Separar características (X) y etiquetas (y)
X = df.drop('Heart_Attack_Risk', axis=1)
y = df['Heart_Attack_Risk']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("El tamaño de X_train es: ", X_train.shape)
print("El tamaño de X_test es: ", X_test.shape)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el modelo de regresión logística
model = LogisticRegression(max_iter=1000, random_state=42)

# Random forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Accuracy (Random Forest):", accuracy_score(y_test, y_pred_rf))

# Entrenar el modelo
model.fit(X_train, y_train)
print("El modelo ha sido entrenado con éxito.")

# Predecir los valores del conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# Visualizar la matriz de correlación
plt.figure(figsize=(15, 10)) 
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title("Matriz de correlación", fontsize=16)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.show()

