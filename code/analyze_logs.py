import re
import matplotlib.pyplot as plt
import numpy as np
l = []
with open('../data/logs/logs_2.txt') as f:
    for line in f:
        x = re.findall(r'\d+\.\d+(?:e-?\d+)?', line)
        if len(x) != 4:
            continue
        l.append(x)

a = np.array(l).astype(np.double)
print(a)
plt.plot(a[:,0])
plt.plot(a[:,2])
plt.ylim([0,0.2])
plt.legend(['Train Loss', 'Val loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(a[:,1])
plt.plot(a[:,3])
plt.ylim([0.8,1])
plt.legend(['Train Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

v_acc = a[:,3]
print(np.argmax(v_acc))
print(len(v_acc))