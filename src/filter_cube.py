import numpy as np
import sys
import argparse
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sigma", type=float, default=2.,
                    help="3D smoothening sigma.")
parser.add_argument("-t", "--threshold", type=float,  default=0.5,
                    help="Threshold for continuum signal masking.")
parser.add_argument("--mask_grow", type=float, default=3.,
                    help="Size for growing the continuum signal mask.")
parser.add_argument('-i', '--infile', type=str, metavar='infile',
                    help='Input cube.')
parser.add_argument('-o', '--outfile', type=str, metavar='outfile',
                    help='Output cube.')
args = parser.parse_args(sys.argv[1:])


hdu = fits.open(args.infile)


# build continuum signal mask
# compute median in spectral direction 
m = np.median(hdu[0].data, axis=0)
# initial mask
zero_mask = m == 0.

mask = np.array( m > args.threshold, dtype=float)

# grow the mask
r = int( np.ceil( args.mask_grow/2.) )
xx = np.arange(-r-1,r+2,dtype=float)
X,Y = np.meshgrid(xx,xx)
dd = np.sqrt( X **2. + Y **2.)
kernel = np.array( dd < r, dtype=float)
kernel = kernel/np.sum(kernel)
smask = fftconvolve(mask, kernel, mode='same')
smask = smask > .1

for i in range(hdu[0].data.shape[0]):
    md = np.median(hdu[0].data[i][ ~zero_mask  * ~smask   ])
    hdu[0].data[i][smask] = md

hdu[0].data = gaussian_filter(hdu[0].data, args.sigma)

for i in range(hdu[0].data.shape[0]):
    hdu[0].data[i][smask] = 0.

hdu.writeto(args.outfile, overwrite=True)

