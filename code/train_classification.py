from pathlib import Path

from keras import Model
from keras.layers import Dense, Flatten, Dropout
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle


K=92 #number of classes
tr_img = 16746
test_img = 4298
gen = ImageDataGenerator(rotation_range = 20,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         shear_range = 0.1,
                         zoom_range = 0.2,
                         horizontal_flip = False,
                         preprocessing_function = tf.keras.applications.resnet_v2.preprocess_input)
batch_size = 256

train_generator = gen.flow_from_directory('../datasets/signClassificationData/train', shuffle = True, target_size = (64,64), batch_size = batch_size)
val_generator = gen.flow_from_directory('../datasets/signClassificationData/test', shuffle = True, target_size = (64,64), batch_size = batch_size)
resnet = tf.keras.applications.ResNet50V2(input_shape=(64,64,3), include_top=False)
x = resnet.output
x = Flatten()(x)
x = Dropout(0.1)(x)
output = Dense(K, activation='softmax')(x)
model = Model(inputs = resnet.input, outputs = output)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=66,
    decay_rate=0.9
)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

cb = tf.keras.callbacks.ModelCheckpoint(
    filepath = 'training/cp-{epoch:04d}.ckpt',
    verbose=1,
    save_weights_only=True,
    save_freq="epoch"
)
r = model.fit(train_generator,
              validation_data = val_generator,
              epochs = 240,
              steps_per_epoch = int(np.ceil(tr_img/batch_size)),
              validation_steps = int(np.ceil(test_img/batch_size)),
              callbacks = [cb])

plt.plot(r.history['loss'] , color = 'red' , label = 'loss')
plt.plot(r.history['val_loss'] , color = 'blue' , label = 'val_loss')

file = Path(f'../training/history')
with file.open("wb") as f:
    print(f'Creating history')
    pickle.dump(r.history, f)

plt.show()
