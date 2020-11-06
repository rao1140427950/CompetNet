import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
from CompetNet import CompetNet


x1 = [cos(pi / 4), sin(pi / 4)]
x2 = [cos(pi / 2), sin(pi / 2)]
x3 = [cos(13 * pi / 18), sin(13 * pi / 18)]
x4 = [cos(pi), sin(pi)]
x5 = [cos(-3 * pi / 4), sin(-3 * pi / 4)]
x = [x1, x2, x3, x4, x5]
x = np.array(x, dtype=np.float32)
x = np.transpose(x)
init_w = np.array([[1, 0], [0, -1]], dtype=np.float32)

net = CompetNet(2, 2, init_w)

for _ in range(256):
    for n in range(5):
        xdata = np.reshape(x[:, n], (2, 1))
        net.train(xdata, 0.1)

rst = net.predict(x)
w = net.get_weights()
plt.scatter(w[0, 0], w[1, 0], c='b', marker='*')
plt.scatter(w[0, 1], w[1, 1], c='r', marker='+')
for n in range(5):
    plt.scatter(x[0, n], x[1, n], c='b' if rst[n] == 0 else 'r')
plt.show()

