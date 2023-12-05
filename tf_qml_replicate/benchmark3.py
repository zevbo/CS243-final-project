from io import TextIOWrapper
import math
import numpy as np
import random
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(1)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(1)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)




ModelT = tf.keras.Model

QUANTIZE = False
def benchmark_3_model() -> ModelT:
    f =  tf.keras.layers.Flatten(input_shape=(28, 28))
    d1 = tf.keras.layers.Dense(units=128, activation='relu')
    d2 = tf.keras.layers.Dense(units=10, activation='linear')
    if QUANTIZE: 
        d1 = tfmot.quantization.keras.quantize_annotate_layer(d1)
        d2 = tfmot.quantization.keras.quantize_annotate_layer(d2)

    model = tf.keras.models.Sequential([f, d1, d2])
    op = tf.keras.optimizers.legacy.SGD(learning_rate=0.02)
    # op = tf.keras.optimizers.Adam(0.001)
    if QUANTIZE:
        model = tfmot.quantization.keras.quantize_apply(model)
    model.compile(optimizer=op, 
        #loss='mean_squared_error'
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

def test_loss(model: ModelT) -> float:
    input_tensor = np.array([[0.3971, 0.7544, 0.5695, 0.4388, 0.6387]])
    output = model(input_tensor)[0].numpy()[0]
    value = function(input_tensor[0])
    loss = (value - output) * (value - output)
    print(f"Test loss {loss}: {value = }, {output = }")

def calc_loss(model: ModelT) -> float:
    loss = 0
    for v in ds_test:
        if loss == 0:
            print(v[0].shape)
            print(v[1].shape)
            loss = 1
        
        # input_tensor = rand_input()   
        # output = model(input_tensor)[0].numpy()[0]
        # value = function(input_tensor[0])
        # loss += (value - output) * (value - output)
    return loss

def train_model(model: ModelT) -> None:
    print(f"Loss at start: {calc_loss(model)}")
    model.fit(ds_train, batch_size= 1, epochs=1)
    # print(f"Loss at end: {calc_loss(model, 1000)}")

def run_benchmark() -> None:
    np.random.seed(0)
    tf.random.set_seed(0)
    tf.keras.utils.set_random_seed(0)
    model = benchmark_3_model()
    # print(f"{len(ds_test) = }")
    # print(f"{d.weights = }")
    # d.set_weights([w,  np.array([0.2682017])])
    # test_loss(model)
    # d.set_weights([[0.23043269, -0.19739035, -0.08669749,  0.20990819, -0.42102337]])
    # [0.2682017]
    # l = calc_loss(model, 100)
    # print(f"{l = }")
    train_model(model)
    # model.compute_loss(i, np.array([function(i)]))
    # print(model(example))

if __name__ == "__main__":
    run_benchmark()
