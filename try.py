import matplotlib.pyplot as plt
import numpy as np

truths = np.ones((18, 128, 128, 3)) * 255.0
fakes = np.zeros((18, 128, 128, 3)) * 255.0

add = []
temp_t = []
temp_f = []
for ind, (t, f) in enumerate(zip(truths, fakes)):
    temp_t.append(t)
    temp_f.append(f)
    if len(temp_f) == 6:
        add.append(temp_t)
        add.append(temp_f)
        temp_t = []
        temp_f = []
output = np.array(add).reshape(6*128,6*128,3)

plt.imshow(output)
plt.show()
