import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
  #tf.keras.layers.Flatten(input_shape=(28, 28)),
  layers.Flatten(input_shape=(28, 28)),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

array([[ 0.67086774, -0.25231966,  0.01695401, -0.20872438, -0.5840499 ,
         0.20415965, -0.07967779,  0.01230302,  0.2564202 ,  0.19890268]],
      dtype=float32)

tf.nn.softmax(predictions).numpy()

array([[0.18120685, 0.07198457, 0.09422877, 0.07519217, 0.05166196,
        0.11362814, 0.08554938, 0.09379152, 0.11972431, 0.11303235]],
      dtype=float32)