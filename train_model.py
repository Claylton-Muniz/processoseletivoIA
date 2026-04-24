import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Carregando o dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizando os dados
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Redimensionando para incluir o canal de cor em escalas de cinza
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Construindo um modelo CNN
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"), # Usando só 64 para maior eficiência 
    layers.Dense(10, activation="softmax"),
])

# Compilando o modelo
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Treinamento limitado a 5 épocas
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Avaliação detalhada
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# Fazendo predições para análise detalhada
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
cm = tf.math.confusion_matrix(y_test, predicted_labels)

# Metricas individuais por classe
diag = np.diag(cm)
precision = np.divide(diag, np.sum(cm, axis=0), out=np.zeros_like(diag, dtype=float), where=np.sum(cm, axis=0) != 0)
recall = np.divide(diag, np.sum(cm, axis=1), out=np.zeros_like(diag, dtype=float), where=np.sum(cm, axis=1) != 0)
f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(diag, dtype=float), where=(precision + recall) != 0)

# Tirando a média das métricas para uma visão geral
precision = np.mean(precision)
recall = np.mean(recall)
f1 = np.mean(f1)

print("\n" + "=" * 50)
print(f"Acurácia: {train_acc:.4f}")
print(f"Acurácia de validação: {val_acc:.4f}")
print(f"Acurácia no Teste: {test_acc:.4f}")
print(f"Perda (Loss) no Teste: {test_loss:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Revocação: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Nota: Acurácia acima de {test_acc * 100:.2f}% com modelo compacto.")
print("=" * 50)

# Salvando o modelo
model.save("model.h5")
print("Modelo salvo!!")
