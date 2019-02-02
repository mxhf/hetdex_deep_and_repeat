
from astropy.stats import biweight_midvariance, biweight_location
from astropy.io import fits
from matplotlib import pyplot as plt
import os
from scipy.signal import fftconvolve
import numpy as np

import sys

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threshold", type=float,  default=0.03,
                    help="Threshold for continuum signal masking.")
parser.add_argument("--mask_grow", type=float, default=10.,
                    help="Size for growing the continuum signal mask.")
parser.add_argument('-i', '--infile', type=str, metavar='infile',
                    help='Input cube.')
parser.add_argument('-o', '--outfile', type=str, metavar='outfile',
                    help='Output cube.')
args = parser.parse_args(sys.argv[1:])

hdu = fits.open(args.infile)


mask_grow = args.mask_grow
threshold = args.threshold
fcube = args.infile
fnew = args.outfile

c = fits.getdata(fcube)
header = fits.getheader(fcube)

# In[61]:


# build continuum signal mask
# compute median in spectral direction 
m = np.nanmedian(c, axis=0)
# initial mask
zero_mask = m == 0.

mask = np.array( m > threshold, dtype=float)

# grow the mask
r = int( np.ceil(mask_grow/2.) )
xx = np.arange(-r-1,r+2,dtype=float)
X,Y = np.meshgrid(xx,xx)
dd = np.sqrt( X **2. + Y **2.)
kernel = np.array( dd < r, dtype=float)
kernel = kernel/np.sum(kernel)
smask = fftconvolve(mask, kernel, mode='same')
smask = smask > .1


# In[62]:


mm = np.zeros(c.shape[0])
ll = np.zeros(c.shape[0])
ss = np.zeros(c.shape[0])
vv = np.zeros(c.shape[0])
for i in range(c.shape[0]):
    nanmask = ~ np.isnan(c[i])
    vv[i] = np.sqrt( biweight_midvariance( c[i][smask * nanmask] ) )
    ss[i] = np.std( c[i][smask * nanmask] )
    mm[i] = np.mean(c[i][smask * nanmask] )
    ll[i] = biweight_location(c[i][smask * nanmask] )

# In[63]:


zz = np.arange(c.shape[0])
ii = ~np.isnan(vv)
p = np.polyfit(zz[ii], vv[ii], deg = 17)


# In[64]:


from scipy.interpolate import UnivariateSpline
spl = UnivariateSpline(zz[ii], vv[ii], k=5, s=2.)


# In[65]:


f = plt.figure(figsize=[15,10])
plt.subplot(2,1,1)
plt.plot(mm)
plt.plot(ll)
plt.ylim([-.1,.1])
plt.grid()

plt.subplot(2,1,2)
plt.plot(ss)
plt.plot(vv)
plt.plot(spl(zz))
plt.ylim([0.,1.])
plt.grid()



# In[66]:


from numpy import random
random.seed(42)
nc = np.zeros_like(c)
N = nc.shape[1] * nc.shape[2]
for i,z in enumerate(zz):
    ns = random.normal(scale = spl(zz[i]), size = N)
    nc[i] = ns.reshape(nc[0].shape)

# In[67]:



# In[69]:


w = 31
m = 9
xc,yc = 48.,65.

ss = []
for i in range(c.shape[0]):

    subim =  c[i,int(yc-w/2):int(yc+w/2),int(xc-w/2):int(xc+w/2)]
    M = []
    for j in range(0,w-m):
        for k in range(0,w-m):
            M.append(subim[j:j+m,k:k+m].flatten() )
    M = np.array(M)

    cov = np.cov(M.T)
    
    ss.append( cov[m**2//2].reshape([m,m]) )
    
ss=np.array(ss)



# In[70]:


w = 20
i = 45
plt.imshow(  np.mean(ss[i*w:i*w+w], axis=0) )


# In[71]:


kernel = np.mean(ss, axis=0)
kernel = kernel/np.sum(kernel)


# In[72]:


from scipy.signal import fftconvolve


# In[73]:


cnc = fftconvolve(kernel,nc[0])


# In[74]:


plt.imshow(cnc)


# In[109]:


maxiter = 15
f = 0.5
DEBUG = True

from numpy import random
random.seed(42)
nc = np.zeros_like(c)
N = nc.shape[1] * nc.shape[2]
for i,z in enumerate(zz):
    #i = 100
    #print(i)
    #i = 500
    # this is the target sigma we need to reach 
    # after taking the covariance into account
    target_sigma = spl(zz[i]) 
    scale = target_sigma * 3.53
    
    iter = 0
    while True:
        if DEBUG:
            print("Iteration {}".format(iter))
            print("target_sigma {}".format(target_sigma))
        if iter > maxiter:
            break
        # generate random noise image (w/o) covaraince
        ns = random.normal(scale = scale, size = N)
        ns = ns.reshape(nc[0].shape)
        # convolve with interpical covariance kernel
        cns = fftconvolve(ns, kernel, mode='same')
        # compute resulting standard deviation
        s = np.std(cns)
        if DEBUG:
            print("Scale {}".format(scale))
            print("sigma {}".format(s))
            print("scale/target_sigma = {}".format(scale/target_sigma))
            print("sigma/target_sigma = {}".format(s/target_sigma))
        if 0.99 < s/target_sigma < 1.01:
            break
        # adjust noise scale by relative difference
        dscale =  scale / (s/target_sigma) - scale
        if DEBUG:
            print("dscale {}".format(dscale))
        scale = scale + f * dscale
        if DEBUG:
            print("New scale {}".format(scale))
            print("")

        iter += 1
    nc[i] = cns
    #break


# In[110]:


plt.imshow(nc[-30])


# In[111]:


h = fits.PrimaryHDU(nc, header=header)
#for i in range(h.data.shape[0]):
#    h.data[i][c[0] == 0.] = 0.

h.writeto(fnew, overwrite=True)



# In[ ]:




