import h5py
import numpy as np    
import matplotlib.pyplot as plt
import os
import matplotlib
import argparse
import tqdm


import scipy.ndimage.filters as filters

def smooth( x, window_len=51, window='hanning'):
    return filters.convolve1d(x, np.ones(window_len)/window_len)

def read_simu(datafolder, type):
    if type == 'prtl':
        os.system('ls '+ os.path.join(datafolder, 'particles/*h5') + ' > ' + os.path.join(datafolder, 'filelist_%s.txt')%type)
        with open(os.path.join(datafolder, 'filelist_%s.txt'%type)) as f:
            files = f.readlines()
            for i in range(len(files)):
                files[i] = files[i].strip()
    elif type == 'flds':
        os.system('ls '+ os.path.join(datafolder, 'fields/*h5') + ' > ' + os.path.join(datafolder, 'filelist_%s.txt')%type)
        with open(os.path.join(datafolder, 'filelist_%s.txt'%type)) as f:
            files = f.readlines()
            for i in range(len(files)):
                files[i] = files[i].strip()
    return files

def plot_pressure(datafolder, ax,ax2, coeff, mi,x=-1, colori = 'gray', colore = 'black', type='flds', label=''):
    files = read_simu(datafolder, type)
    file = h5py.File(files[x],'r+')['Step0']
    T22_e = np.array(file['fT22_1'])
    T11_i = np.array(file['fT11_2'])
    T22_i = np.array(file['fT22_2'])
    x1 = np.array(file['X1'])

    ax.plot(x1, coeff + smooth(T22_e / T11_i[-600:-500].mean() / mi**2), lw = 1,color = colore, label=label)

    ax2.plot(x1, coeff + smooth(T22_i / T11_i[-600:-500].mean()),  lw = 1,color = colori, label=label)# + '$+$' + str('%.1f'%coeff))
    # ax.set_yscale('log')
    ax.set_xlabel(r'x $[c / \omega_i]$')
    ax.set_ylabel(r'$ P / P_{i,\infty}$')
    return 


def plot_B3(datafolder, ax,x=-1, color = 'gray', type='flds'):
    files = read_simu(datafolder, type)
    file = h5py.File(files[x],'r+')['Step0']
    B3 = np.array(file['fB3'])
    x1 = np.array(file['X1'])/5

    ax.scatter(x1, B3,s = 0.01, color = color)
    ax.set_yscale('log')
    ax.set_xlabel(r'x $[c / \omega_i]$')
    ax.set_ylabel(r'$B_3$')
    return 

def plot_density(datafolder, ax, coeff, x=-1, colori= 'gray', colore = 'black', type='flds', label=''):
    files = read_simu(datafolder, type)
    file = h5py.File(files[x],'r+')['Step0']
    x1 = np.array(file['X1'])/5
    n1 = np.array(file['fN_1'])
    n2 = np.array(file['fN_2'])
    ax.plot(x1, smooth(n1) + coeff, lw = 0.6, color = colore, label=label + '$ + $' + str(coeff))

    ax.plot(x1, smooth(n2) + coeff, lw = 0.6, color = colori)
    ax.set_xlabel(r'x $[c / \omega_i]$')
    ax.set_ylabel(r'N')

    return   

def plot_ux(datafolder, x=-1, type='prtl'):
    def getuw(x,Xsh, Lsh, driftux):
        return (5./8. + 3./8. * np.tanh(2 * (x - Xsh)/Lsh)) * ( - driftux)
    files = read_simu(datafolder, type)
    file = h5py.File(files[x],'r+')['Step0']
    u11 = np.array(file['pU1_1'])
    u12 = np.array(file['pU1_2'])
    x1 = np.array(file['pX1_1'])/5
    x2 = np.array(file['pX1_2'])/5

    # xmax = x2.max()
    xmin,xmax = 0,x1.max()
    fill=0.6

    plt.figure()
    plt.hist2d(x1, u11, bins=(np.linspace(xmin,xmax,1001),np.linspace(-0.5,0.5,201)), norm=matplotlib.colors.LogNorm())
    plt.ylabel('ux')
    plt.xlabel(r'x $[c / \omega_i]$')
    plt.plot([0,xmax],[-0.1,-0.1], color = 'red', lw=0.5)
    plt.colorbar()
    plt.plot(np.linspace(xmin,xmax,1001), getuw(np.linspace(xmin,xmax,1001),x1.max()*fill, 100, 0.1), color = 'orange', label = r'$u_W$')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.hist2d(x2, u12, bins=(np.linspace(xmin,xmax,2001),np.linspace(-0.2,0.2,201)), norm=matplotlib.colors.LogNorm())
    plt.ylabel('ux')
    plt.xlabel(r'x $[c / \omega_i]$')
    plt.plot([xmin,xmax],[-0.1,-0.1], color = 'red', lw=0.5)
    plt.plot(np.linspace(xmin,xmax,1001), getuw(np.linspace(xmin,xmax,1001),x1.max()*fill, 100, 0.1), color = 'orange', label = r'$u_W$')
    plt.colorbar()
    plt.legend()
    return

def plot_spec(datafolder, ax, c1, spec,mi, range,x=-1, type='prtl', label='',nbins=50, coeff=1):
    files = read_simu(datafolder, type)
    file = h5py.File(files[x],'r+')['Step0']
    time = file['Time']
    if spec == 1:
        specname = 'e'
    elif spec == 2:
        specname= 'i'
    def integrate(x,y):
        return ((y[1:]+ y[:-1])/2*(x[1:] - x[:-1])).sum()

    def plotvdis(xlow, xhigh, spec, file, label, color, ax, coeff):
        xu,yu = AngDist(xlow, xhigh,spec,file, mi,nbins)
        ax.plot(xu,smooth(yu, window_len=3)*xu* coeff, label=label, color = color, lw=1.2)
        return integrate(xu,yu*xu) /integrate(xu,yu) #xu[np.where(yu*xu == (yu*xu).max())]

    ted = plotvdis(*range,spec,file,label, c1,ax,coeff)
    # ted = plotvdis(*range,spec,file,label+'$%c_{down}$'%specname, c1,ax)

    # ax.set_xscale('log')
    # ax.set_yscale('log')

    # ax.set_ylabel(r'$\gamma dN/d\gamma$',fontsize = 15)
    # ax.set_xlabel(r'$E_{kin} [m_e c^2]$',fontsize = 15)
    return time

# def plot_spec_evo()

def AngDist(x1, x2, spec, file,mi, nbins):
    if spec == 1:
        m = 1
    else:
        m = mi
    spec = str(spec)
    x = np.array(file['pX1_' + spec])
    u1 = np.array(file['pU1_'+ spec])
    u2 = np.array(file['pU2_'+ spec])
    u3 = np.array(file['pU3_'+ spec])

    targidx = np.argwhere((x1< x) & (x < x2)).flatten()
    u1 = u1[targidx]
    u2 = u2[targidx]
    u3 = u3[targidx]

    v1 = u1 / np.sqrt(1 + u1**2 + u2**2 + u3**2)
    v2 = u2 / np.sqrt(1 + u1**2 + u2**2 + u3**2)
    v3 = u3 / np.sqrt(1 + u1**2 + u2**2 + u3**2)

    V1 = v1.mean()
    V2 = v2.mean()
    V3 = v3.mean()
    V = np.array([V1, V2, V3])
    V2_mag2 = np.dot(V, V)

    v_particle = np.stack([v1, v2, v3], axis=1)

    dot = np.sum(v_particle * V, axis=1)
    factor = 1 / (1 - dot)
    v_rel = (v_particle - V) * factor[:, None]

    v_mod = np.linalg.norm(v_rel, axis=1)

    gm = 1 / np.sqrt(1 - v_mod**2)

    v_mod = (gm-1) * m

    bins = np.logspace(np.log10(v_mod.min() + 1e-8), np.log10(v_mod.max()), nbins)
    img = np.histogram(v_mod, bins=bins, density=False)
    # plt.close()
    
    return (img[1][:-1], img[0][:])


def integrate(x,y):
    return ((y[1:]+ y[:-1])/2*(x[1:] - x[:-1])).sum()

def stepintegrate(x,y):
    res = np.zeros_like(y)
    for i in range(1, res.shape[0]-1):
        res[i] = integrate(x[i:], y[i:])
    return res

def plot_potential(datafolder, ax,x=-1, color = 'gray', type='flds', label = None, mr=25):
    x1 = None
    # a = 1
    # for idx in tqdm.trange(x-a,x):
    files = read_simu(datafolder, type)
    file = h5py.File(files[x],'r+')['Step0']
    Exmean = np.array(file['fAvgEx'])/10
    T11 = np.array(file['fT11_2'])
    if x1 is None:
        x1 = np.array(file['X1'])
    else:
        x1 += np.array(file['X1'])
    # x1 =  x1 / a
    ax.plot(x1, stepintegrate(x1, Exmean)/ (T11[-600:-500].mean() * mr**2), color=color, label=label)
    # ax.set_yscale('log')
    ax.set_xlabel(r'x $[c / \omega_i]$')
    ax.set_ylabel(r'$ \Phi_E$')
    return 

