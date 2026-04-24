import tensorflow as tf
import os

# Carregando o modelo treinado
model = tf.keras.models.load_model("model.h5")

# Convertendo o modelo para o formato TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Ativando otimizações
tflite_model = converter.convert()

# Salvando o modelo otimizado
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo otimizado salvo!!")
