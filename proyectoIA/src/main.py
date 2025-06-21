import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV



# 1. Comprensión del negocio
# Objetivo: Determinar si una reseña de película es positiva o negativa.
# Beneficio: Ayuda a empresas de cine o plataformas de streaming a entender la opinión del público.

# 2. Entendimiento de los datos
# Carga del dataset
df_train = pd.read_csv("C:/Users/Dragu/PycharmProjects/proyectoIA/src/archive/Train.csv")
df_test = pd.read_csv("C:/Users/Dragu/PycharmProjects/proyectoIA/src/archive/Test.csv")
df_valid = pd.read_csv("C:/Users/Dragu/PycharmProjects/proyectoIA/src/archive/Valid.csv")

# Unir los datasets en uno solo
df_review = pd.concat([df_train, df_test, df_valid], ignore_index=True)

# Mostrar información del dataset
print(df_review.info())
print(df_review.head())
# Ver nombres de las columnas antes de usarlas
print(df_review.columns)
print(df_review['label'].value_counts())

# 3. Preparación de los datos
# Balanceo de datos usando RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_bal, y_bal = rus.fit_resample(df_review[['text']], df_review['label'])

# Convertimos a DataFrame
df_review_bal = pd.DataFrame({'review': X_bal['text'], 'sentiment': y_bal})

print(df_review_bal['sentiment'].value_counts())

# División en train y test
train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

# Vectorización TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
test_x_vector = tfidf.transform(test_x)

# 4. Modelado
models = {
    "SVM": SVC(kernel='linear'),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression()
}

for name, model in models.items():
    model.fit(train_x_vector, train_y)  # Entrenamos el modelo
    accuracy = model.score(test_x_vector, test_y)  # Evaluamos el modelo
    print(f"{name} Accuracy: {accuracy:.4f}")

# 5. Evaluación
svc = models["SVM"]
predictions = svc.predict(test_x_vector)
print(classification_report(test_y, predictions))

# Matriz de confusión
conf_mat = confusion_matrix(test_y, predictions, labels=[1, 0])
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=['positive', 'negative'], yticklabels=['positive', 'negative'])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# 6. Optimización con GridSearchCV
params = {'C': [1, 4, 8, 16, 32], 'kernel': ['linear', 'rbf']}
svc_grid = GridSearchCV(SVC(), params, cv=5)
svc_grid.fit(train_x_vector, train_y)
print(svc_grid.best_params_)
print(svc_grid.best_estimator_)
