import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Carregando o dataset MNIST e separando em conjuntos de treinamento e teste
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizando os dados
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Ajustando o formato dos dados para o modelo
# O 28x28 é a dimensão da imagem e o 1 é o número de canais (preto e branco)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Construindo o modelo de rede neural
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

# Compilando o modelo
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Treinando o modelo
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Avaliando o modelo no conjunto de teste
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Teste de acurácia: {test_acc:.4f}")

# Salvando o modelo treinado
model.save("model.h5")
print("Modelo salvo!!")
