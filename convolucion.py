# import numba as nb
import numpy as np


def correlacion(signal, kernel):
    ker_size = kernel.size
    sig_size = signal.size
    pad_size = (ker_size - 1) // 2
    padded_signal = np.pad(signal, pad_size, mode='constant')
    output = np.zeros(sig_size)
    for k in range(sig_size):
        print(padded_signal[k:k+ker_size], kernel)
        output[k] = np.dot(padded_signal[k:k+ker_size], kernel)
    return output


def convolveDot(signal, kernel):
    ker_size = kernel.size
    sig_size = signal.size
    kernel = kernel[::-1]
    pad_size = (ker_size - 1) // 2
    padded_signal = np.pad(signal, pad_size, mode='constant')
    output = np.zeros(sig_size)
    for k in range(sig_size):
        print(padded_signal[k:k+ker_size], kernel)
        output[k] = np.dot(padded_signal[k:k+ker_size], kernel)
    return output


def convolvef(signal, kernel):
    output = np.zeros(len(signal))
    kernel = kernel[::-1]
    m = len(kernel) // 2
    for n in range(len(signal)):
        for k in range(-m, m+1):
            if n + k < 0 or n + k >= len(signal):
                continue
            print(n,k,signal[n+k], kernel[k+m])
            output[n] += signal[n+k] * kernel[k+m]
            # print( output)
    return output



# x = np.array([1, -3, 0, 4, 5, 6])
# h = np.array([0, 1, -1])
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# x = np.ones(10)*8
print(x)
h=np.array([1, 2, 1])



print("Dot product convolution: ")
print(convolveDot(x, h))
# print("for loop convolution: ")
# print(convolvef(x, h))



# print("Numpy convolution: ")
# print(np.convolve(x, h, mode='same'))


# print("correlacion: ")
# print(correlacion(x, h))
# print("Numpy correlate: ")
# print(np.correlate(x, h, mode='same'))


def convolucion4F(img, kernel):
  """
  Regresa imagen con operación de ventana, 
  args: imagengray y matriz kernel 
  """
  kernel = np.flip(kernel)
  nr = img.shape[0]
  nc = img.shape[1]

  nk = kernel.shape[0]//2
  nl = kernel.shape[1]//2

  out = np.zeros(img.shape)

  # con 4 ciclos for
  for r in range(nr):
    for c in range(nc):
      for k in range(-nk, nk+1):
        for l in range(-nl, nl+1):  # corrected indexing
          if r-k >= 0 and c-l >= 0 and r-k < nr and c-l < nc:
            out[r, c] += kernel[k+nk, l+nl]*img[r-k, c-l]
  return out


img = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

out = convolucion4F(img, kernel)
print(out)
