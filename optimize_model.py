import tensorflow as tf
import numpy as np

# Carregando o modelo treinado
model = tf.keras.models.load_model("model.h5")

# Dataset representativo visando fazer uma otimização chamada "Full Integer Quantization"
def representative_data_gen():
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    # 100 amostras para calibração, normalizadas como no treino
    for i in range(100):
        image = x_train[i].astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=(0, -1))
        yield [image]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Otimização padrão
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Dataset representativo para quantização completa
converter.representative_dataset = representative_data_gen

# Garante que as operações sejam convertidas para INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32  # Mantém entrada flexível
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

# Salvando o modelo otimizado
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo otimizado e salvo!!")
