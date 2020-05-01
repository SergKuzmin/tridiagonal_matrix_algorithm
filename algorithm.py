import numpy as np


def t_m_a(a: np.array, b: np.array, c: np.array, f: np.array) -> np.array:
    i_shape = b.shape
    if a.shape[0] == i_shape[0] - 1:
        a = np.concatenate([a, [np.zeros(shape=i_shape[1:])]])
    elif (a.shape[0] < i_shape[0] - 1) or (a.shape[0] > i_shape[0]):
        raise Exception("(a.shape[0]) must be equal (b.shape[0]) or equal (b.shape[0] - 1)")

    if c.shape[0] == i_shape[0] - 1:
        c = np.concatenate([[np.zeros(shape=i_shape[1:])], c])
    elif (c.shape[0] < i_shape[0] - 1) or (c.shape[0] > i_shape[0]):
        raise Exception("(c.shape[0]) must be equal (b.shape[0]) or equal (b.shape[0] - 1)")

    a = a.astype(np.float)
    b = - b.astype(np.float)
    c = c.astype(np.float)
    f = - f.astype(np.float)

    y = np.zeros(shape=i_shape[:2]).astype(np.float)
    z = np.zeros(shape=i_shape[:2]).astype(np.float)
    x = np.zeros(shape=i_shape).astype(np.float)

    x[1] = (np.linalg.inv(b[0])).dot(a[0])
    for i in range(1, i_shape[0] - 1):
        x[i + 1] = (np.linalg.inv((b[i] - (c[i]).dot(x[i])))).dot(a[i])

    z[1] = (np.linalg.inv(b[0])).dot(f[0])
    for i in range(1, i_shape[0] - 1):
        z[i + 1] = (np.linalg.inv((b[i] - (c[i]).dot(x[i])))).dot((c[i].dot(z[i]) + f[i]))

    y[i_shape[0] - 1] = \
        (np.linalg.inv((b[i_shape[0] - 1] - (c[i_shape[0] - 1]).dot(x[i_shape[0] - 1])))). \
            dot((c[i_shape[0] - 1].dot(z[i_shape[0] - 1]) + f[i_shape[0] - 1]))

    for i in range(i_shape[0] - 2, -1, -1):
        y[i] = (x[i + 1].dot(y[i + 1])) + z[i + 1]

    return y
