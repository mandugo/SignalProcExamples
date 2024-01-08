import matplotlib.pyplot as plt
import numpy as np

T = 10
f0 = 1.8
fs = 2
Ts = 1/fs

n = np.arange(0,T,Ts)
N = len(n)

signal = np.exp(1j*2*np.pi*f0*n)# + np.random.normal(0, 1, N)

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(n, np.real(signal), color='red')
plt.xlabel('t (s)')
plt.ylabel('Amplitude')
plt.title(rf'$T = {T}\, s, \enspace N = {N}$')
plt.grid(True)

window = np.hamming(N)
winSig = window*signal

Nfft = int(np.power(2, np.ceil(np.log2(N))))
spectrum = np.abs(np.fft.fft(winSig, Nfft))
fAx = np.linspace(0, fs, Nfft)

indMax = np.argmax(spectrum)
peak = fAx[indMax]

lower_bound = (fAx[indMax] - fs/N)%fs
upper_bound = (fAx[indMax] + fs/N)%fs

plt.subplot(122)
plt.plot(fAx, spectrum, color='blue')
plt.stem(fAx[indMax],spectrum[indMax],'go')
plt.title(rf'$N_{{fft}} = {Nfft}, \enspace f_{{max}} = {np.round(fAx[indMax],2)} \, Hz$')
plt.axvline(x=lower_bound, color='m', linestyle='--', label='Lower Bound')
plt.axvline(x=upper_bound, color='m', linestyle='-.', label='Upper Bound')
plt.grid(True)
plt.xlabel('f (Hz)')
plt.ylabel('Amplitude Spectrum')
plt.suptitle(rf'$f_{{0}} = {f0} \, Hz, \enspace f_{{s}} = {fs} \, Hz$', fontsize=16)
plt.legend()

plt.show()