# build catalog

from astropy.table import Table

import spectrum
from collections import OrderedDict
import argparse
from astropy.io import fits
from astropy import wcs
import argparse
import sys
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--incube', type=str,                    
                    help='Input cube.')
parser.add_argument('-m', '--map', type=str,                    
                    help='Input map.')
parser.add_argument('-o', '--output_catalog', type=str, default='',
                    help='Output catalog.')
args = parser.parse_args(sys.argv[1:])


names   = [ "id", "N", "flux", "ra_com", "dec_com", "ddec", "dra", "x_com", "y_com", "z_com", 
           "dx", "dy", "dz", 
           "diam",
           "x_ext", "y_ext", "z_ext",
           "wl_com", "dwl", "xmin","xmax", "ymin", "ymax", "zmin", "zmax"]
t = Table(names=names)

dtypes   =  {"id" : int, "N" : int, "flux" : float, 
             "ra_com" : float, "dec_com" : float, 
             "dra" : float, "ddec" : float,
             "x_com": float, "y_com": float, "z_com": float,
             "dx": float, "dy": float, "dz": float,
             "diam": float,
             "x_ext": int, "y_ext": int, "z_ext": int, 
             "wl_com" : float, "dwl": float, 
             "xmin": int,"xmax": int, "ymin": int, "ymax": int, "zmin": int, "zmax": int}

formats  =  {"id" : "5d", "N" : "5d", "flux" : ".4e", "ra_com" : "12.6f", "dec_com" : "12.6f", "dra" : "3.1f", "ddec" : "3.1f", 
             "x_com": "8.2f", "y_com": "8.2f", "z_com": "8.2f", 
             "dx": "8.2f", "dy": "8.2f", "dz": "8.2f", 
             "diam": "8.2f",
             "x_ext": "4d", "y_ext": "4d", "z_ext": "4d", 
             "wl_com" : "8.2f", "dwl": "8.2f", "xmin": "4d","xmax": "4d", "ymin": "4d", "ymax": "4d", "zmin": "4d", "zmax": "4d"}

units    =  {"id" : "", "N" : "px", "flux" : "arb", "ra_com" : "Deg[J2000]", "dec_com" : "Deg[J2000]", "dra" : "arcsec", "ddec" : "arcsec", 
             "x_com": "px", "y_com": "px", "z_com": "px", 
             "dx": "px", "dy": "px", "dz": "px", 
             "diam": "arcsec",
             "x_ext": "px", "y_ext": "px", "z_ext": "px", 
             "wl_com" : "A", "dwl": "A", "xmin": "px","xmax": "px", "ymin": "px", "ymax": "px", "zmin": "px", "zmax": "px"}


s = spectrum.readSpectrum(args.incube)
c = s.data

zz,yy,xx = [np.arange(s, dtype=int) for s in c.shape]
YY,ZZ,XX = np.meshgrid(yy, zz, xx)

w = wcs.WCS(s.hdu)
platescale = s.hdu.header["CDELT2"]
outmap = fits.getdata(args.map)

for n in names:
    t[n].dtype  = dtypes[n]
    t[n].format = formats[n]
    t[n].unit =  units[n]

rr = np.sort( np.unique( outmap.flatten() ) )
# Generate catalog
for r in rr[rr>0]:
    ii = outmap == r
    N = np.sum(ii)
    M = np.sum(c[ii])
    x_com = np.sum(c[ii]*XX[ii])/M
    y_com = np.sum(c[ii]*YY[ii])/M
    z_com = np.sum(c[ii]*ZZ[ii])/M
    
    dx = np.sqrt( np.sum( c[ii] * (XX[ii] - x_com)**2. ) / M ) * 2.35 # FWHM
    dy = np.sqrt( np.sum( c[ii] * (YY[ii] - y_com)**2. ) / M ) * 2.35 # FWHM
    dz = np.sqrt( np.sum( c[ii] * (ZZ[ii] - z_com)**2. ) / M ) * 2.35 # FWHM
    
    diam = np.sqrt(dx**2. + dy**2.)/np.sqrt(2.) * platescale * 3600. # FWHM
    
    
    #wl_com = float(ww_interp(z_com))
    zmax,zmin = (ZZ[ii].max()) , (ZZ[ii].min())
    ymax,ymin = (YY[ii].max()) , (YY[ii].min())
    xmax,xmin = (XX[ii].max()) , (XX[ii].min())
    
    z_ext = zmax - zmin
    x_ext = xmax - xmin
    y_ext = ymax - ymin
    
    # use atropy wcs to convert x,y,z, dz to ra,dec,wl  ... and in bit of a hack ... dwl
    voxel = w.wcs_pix2world([[x_com,y_com,z_com],[x_com,y_com,ZZ[ii].max()],[x_com,y_com,ZZ[ii].min()]],1)
    dwl = voxel[1][2] - voxel[2][2]
    ra_com, dec_com,wl_com = voxel[0]
    
    dra = platescale*dx * 3600.
    ddec = platescale*dy * 3600.
    t.add_row([ r, N, M, ra_com, dec_com, dra, ddec, x_com, y_com, z_com, dx, dy, dz, diam, x_ext, y_ext, z_ext, wl_com, dwl, xmin,xmax, ymin, ymax, zmin, zmax] )
    
t.write(args.output_catalog, format="ascii.ecsv", overwrite=True)