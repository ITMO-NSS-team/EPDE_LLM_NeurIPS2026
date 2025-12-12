import matplotlib.pyplot as plt
import numpy as np

u = np.load('data/noise_level_0/wave/u.npy')
def noise_data(data, noise_level):
    # add noise level to the input data
    return noise_level * np.std(data) * np.random.normal(size=data.shape) + data

# u_ = noise_data(u, 1000)
# plt.imshow(u_)
# plt.show()

# plt.plot(u[:, 50], color = 'k')
# plt.plot(u_[:, 50], color = 'r')
# plt.show()


data = np.load("data/ds.npy", allow_pickle=True).item()['u'][:, 4]
plt.imshow(data.reshape(u.shape))
plt.show()

d2udt2 = np.load('data/noise_level_0/wave/d^2u_dx1^2.npy')
plt.imshow(d2udt2.reshape(u.shape))
plt.show()

d2udx2u = np.load('data/noise_level_0/wave/d^2u_dx2^2.npy')
plt.imshow(d2udx2u.reshape(u.shape))
plt.show()

# u = np.load('data/noise_level_0/wave/u.npy')
# plt.imshow(data.reshape(u.shape))
# plt.show()