import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fft(arr, fs, t_tot):
    N = int(fs*t_tot) # Nsamples in signaal
    T = t_tot/N # Sampling tijd

    fhat = np.fft.fft(arr)
    x = np.linspace(0.0, 1.0/(2.0*T), N//2) # frequency axis
    y = 2.0/N * np.abs(fhat[:N//2])
    return x,y


def fft_partitioned(arr, fs, t_tot, t_bin=1e-3):
    """
    Voert een FFT uit over arr in windows ter grootte van t_bin.
    
    arr : input array 1D
    fs : sample rate
    t_tot : totale tijdsduur van arr in seconden
    t_bin : gewenste tijdsduur van de bins
    
    return dataframe met 
    t : tijd in seconden
    f : frequentie in Hz
    I : intensiteit in a.u.
    """
    bin_size = fs*t_bin # hoeveelheid samples in 1 bin
    N = arr.shape[0]/bin_size # hoeveelheid bins

    arr_ = np.array_split(arr, N) # output is een list met arrays

    xs = [] # Een lijst die we gaan vullen met x waarden. 's' voor meervoud (plural)
    ys = []
    for array in arr_:
        x,y = fft(arr=array, fs=fs, t_tot=t_bin) # merk op dat t_tot van de array is! t_tot <=> t_bin
        xs.append(x)
        ys.append(y)

    x = np.array(xs) # maak een array van de lijst xs
    y = np.array(ys)
    
    # voeg per bin hier de coordinaat paren aan toe en bereken t.
    dfs = []
    for i in range(x.shape[0]):

        t_ = np.ones(x[0].shape[0])*(i+1)*t_bin # merk op dat we een afrond fout hebben in de uiteindelijke df
        x_ = x[i]
        y_ = y[i]

        d= {'t':t_, 'f':x_, 'I':y_}
        df = pd.DataFrame(data=d) # df of this bin
        dfs.append(df)

    df = pd.concat(dfs)
    return df # nu hebben we een dataframe met coordinaat paren van alles wat we willen hebben.