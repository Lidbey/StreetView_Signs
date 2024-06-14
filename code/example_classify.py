import glob
import tensorflow as tf
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import classification
from data import classification_size, num_classes

SHOW_PER_CLASS = 5

batch_size = 256
gen = classification.get_generator()
val_generator = gen.flow_from_directory('../datasets/signClassificationData/test', shuffle=False,
                                        target_size=classification_size, batch_size=batch_size)
model = classification.get_model('../models/class_models/max_acc2')

Y = model.predict(val_generator)
Y = np.argmax(Y, axis=1)
result = confusion_matrix(val_generator.labels, Y, normalize='pred')
print(result)
ticks = np.linspace(0, num_classes - 1, num=num_classes)
plt.imshow(result, interpolation='none')
plt.colorbar()
#plt.xticks(ticks, fontsize=0)
#plt.yticks(ticks, fontsize=0)
plt.xticks([])
plt.yticks([])
plt.grid(True)
#plt.xticks([])
#plt.yticks([])
#plt.axis('off')
plt.title('Classification confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

i = 0
for folder in glob.glob('../datasets/signClassificationData/test/*'):
    j = 0
    for file in glob.glob(folder + '/*'):
        print(file)
        im = cv2.imread(file)
        im = cv2.resize(im, classification_size)
        X = np.empty((1, *classification_size, 3))
        X[0, ] = im
        cv2.imshow("img", im)
        cv2.waitKey(0)
        X = tf.keras.applications.resnet_v2.preprocess_input(X)
        r = model.predict(X)
        print(np.argmax(r), i)
        j = j + 1
        if j > SHOW_PER_CLASS:
            break
    i = i + 1
