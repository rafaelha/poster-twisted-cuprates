import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
import glob
import os
import re
from scipy import interpolate
from scipy import signal
import scipy.optimize

plt.ion()
#plt.close('all')
files_unsorted = glob.glob('raw_data/*.DAT')
N = len(files_unsorted)
temp = np.array([int(re.findall('\d+', f.replace('raw_data',''))[0]) for f in files_unsorted])
idx = temp.argsort()
temp = temp[idx]
np.savetxt('processed_data/temp.csv', temp)

files = [files_unsorted[i] for i in idx]

def load():
    a, b = np.loadtxt(files[0], unpack=True, skiprows=1)
    L = len(a)*2
    print(a.shape)
    print(b.shape)

    

    X = np.zeros((L,N))
    Y = np.zeros((L,N))
    for i in np.arange(N): 
        x, y = np.loadtxt(files[i], skiprows=1, unpack=True)
        X[0:len(x),i] = x * 1e6
        Y[0:len(y),i] = y 

        
    return X, Y

X, Y = load()

def find_center(x,y):
    m1 = x*1
    m1[x>0] = -10
    m2 = x*1
    m2[x<0] = 10 

    i = m1.argmax()
    j = m2.argmin()
    return y[i] + (y[j] - y[i])/(x[j]-x[i]) * (-x[i]) # return the interpolated value where curve crosses y-axis
def center(dx, x,y):
    yy = y - find_center(x-dx, y)

    f1 = interpolate.interp1d(x-dx, yy)
    f2 = interpolate.interp1d(-(x-dx), -yy)
    xx = np.linspace(-10, 10, 100)
    return np.sum((f1(xx) - f2(xx))**2)


def plot_data(ind, max=None, num=-1):
    x = X[:,ind]
    y = Y[:,ind]
    T = temp[ind]
    cond = np.logical_and(x!=0, y!=0)
    x = x[cond]
    y = y[cond]

    idx = x.argsort()
    x = x[idx]
    y = y[idx]

    plt.plot(x, y, label='T='+str(T)+'mK')
    plt.ylabel('Diff. conductance dI/dV in $\Omega^{-1}$')
    plt.xlabel('Bias in $\mu$V')
    #plt.title('T='+str(T)+'mK')
    cutoff =10 
    #plt.xlim((-cutoff,cutoff))
    plt.tight_layout()
    #plt.axhline(y=0, color='k')
def linear_fit(ind, vmax=40, mirror=False, plot=False):
    x = X[:,ind]
    y = Y[:,ind]
    cond = np.logical_and(x!=0, y!=0)
    x = x[cond]
    y = y[cond]
    # make sure data is sorted
    idx = x.argsort()
    x = x[idx]
    y = y[idx]
    T = temp[ind]
    i = cumtrapz(y, x)
    xi = x[1:]
    # center I have deactivated center function, as it made everything just worse
    res = scipy.optimize.minimize(center, x0=0, args=(xi, i))
    xi -= res.x # correct x-position
    #print(res.x)
    i = i - find_center(xi, i) # correct y-position

    i = i[xi<vmax]
    xi = xi[xi<vmax]
    i = i[xi>-vmax]
    xi = xi[xi>-vmax]

    def flin(x, a):
        return a*x
    coeff, cov = curve_fit(flin, xi, i)
    print('fit:',str(coeff))

    xfit = np.linspace(np.min(xi), np.max(xi), 1000)

    if plot:
        plt.figure(figsize=(4/2,3.6/2))
        plt.plot(xi, i)
        if mirror:
            plt.plot(-xi, -i, '-.')

        #plt.plot(xfit, coeff*xfit, 'r--')
        plt.ylabel('Current I in $\mu A$')
        plt.xlabel('Bias in $\mu$V')
        plt.tight_layout()
        

    return coeff

def fit_coeffs(vmax=40):
    fit_res = np.zeros(len(temp))
    for i in np.arange(len(temp)):
        fit_res[i] = linear_fit(i, vmax=vmax)
    plt.figure(figsize=(4,3.6))
    plt.plot(temp, fit_res, '.-')
    plt.xlabel('T in mK')
    plt.ylabel('Plateau conductance from lin. fit in $\Omega^{-1}$')
    plt.tight_layout()
def plot_int(ind, max=None, cutoff=10, subtract=True, save=False, mirror=False, fitvmax=20):
    T = temp[ind]
    x = X[:,ind]
    y = Y[:,ind]
    cond = np.logical_and(x!=0, y!=0)
    x = x[cond]
    y = y[cond]

    # make sure data is sorted
    idx = x.argsort()
    x = x[idx]
    y = y[idx]

    if subtract:
        y -= linear_fit(ind, vmax=fitvmax)

    """
    plt.figure(figsize=(4,3.6))
    plt.plot(x, y)
    plt.ylabel('dI/dV')
    plt.xlabel('Bias in $\mu$V')
    plt.title('T='+str(T)+'mK')
    plt.ylim((-2,2))
    plt.tight_layout()
    plt.savefig('fig/didv_'+str(num)+'.pdf')
    """
    #integrate
    i = cumtrapz(y, x)
    xi = x[1:]
    res = scipy.optimize.minimize(center, x0=0, args=(xi, i))
    xi -= res.x # correct x-position
    print('Adjusted x-position by', str(res.x))
    i = i - find_center(xi, i) # correct y-position

    #plt.figure(figsize=(4,3.6))
    #plt.plot(xi, i, label='$I(V)$')
    plt.plot(xi, i*1000, label='T='+str(T)+'mK')
    if mirror:
        plt.plot(-xi, -i*1000, label='$-I(-V)$')
    plt.ylabel('Current I in $\mu A$')
    plt.xlabel('Bias in $\mu$V')
    #plt.title('T='+str(T)+'mK')
    plt.xlim((-cutoff,cutoff))
    new_max = np.max(i[xi<cutoff])
    if max == None or max < new_max:
        max = new_max
    #plt.ylim((-0.05,0.05))
    #plt.legend()
    plt.tight_layout()
    #plt.axhline(y=0, color='k')
    #plt.savefig('fig/I_'+str(num)+'.pdf')
    #plt.close('all')

    if ind != -1 and save:
        np.savetxt('processed_data/'+str(ind)+'x.csv', xi)
        np.savetxt('processed_data/'+str(ind)+'y.csv', i)
    return max, xi


plt.figure(figsize=(4,3.6))
for i in [0]:#np.arange(N):
    #_, max, xi = plot_int(X[:,i], Y[:,i], T=temp[i], subtract=True, max=max, num=i)
    plot_int(i, subtract=True, cutoff=15, save=False, fitvmax=20)
    plt.ylim((-38,38))
    plt.ylabel('Supercurrent in nA')
    plt.tight_layout()
    #plot_data(i)
plt.savefig('supercurrent.pdf')

#plt.legend()

linear_fit(1, vmax=70, plot=True)
plt.savefig('liner_contrib_wide.pdf')
