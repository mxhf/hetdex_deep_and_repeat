# spectrum.py v0.2 by Maximilian H. Fabricius
# Generic class to represent a spectrum
#  
from __future__ import print_function

from scipy import inf,log,exp,arange
from scipy import interpolate

from astropy.io import fits
import sys

INFINITY = 1e6

class Spectrum:
    def __init__(self, start, step, data, filename = ""):
        self.start = start
        self.step  = step
        self.data  = data
        self.filename  = filename
        self.hdu = fits.PrimaryHDU()

    def ndim(self):
        return len( self.data.shape )

    # deep copy
    def copy(self):
        return Spectrum(self.start, self.step, self.data.copy())

    def nsteps(self):
        if len( self.data.shape ) == 1:
            return self.data.shape[0]
        elif len( self.data.shape ) == 2:
            return self.data.shape[1]
        elif len( self.data.shape ) == 3:
            return self.data.shape[0]
        else:
            print("ERROR: Don't know how to deal with %d dim data." % len(self.data.shape) )

    def grid(self, start=-INFINITY, stop=INFINITY):
        grid = arange(self.nsteps()) * self.step + self.start
        ii = (grid >= start) * (grid <= stop)
        return grid[ii]

    def extract(self, row, start=-INFINITY, stop=INFINITY):
        if len(self.data.shape) == 1:
            if row == 0:
                grid = self.grid()
                ii = (grid >= start) * (grid <= stop)
                return self.data[ii]
            else:
                print( "ERROR in extract: This sepctrum in one-dimensional but you want to extract row %d." % row)    
        elif len(self.data.shape) == 2:
                grid = self.grid()
                ii = (grid >= start) * (grid <= stop)
                return self.data[row,ii]
        else:
            print( "ERROR in extract: extract method not implemented for spectra with more than 1 dimension." )
            
    def getWL(self,bin,wls, bounds_error=True):
        # returns value for given bin and wavelength, interpolates
        # linearely
        global n, grid, ww
        grid = self.grid()
        ww = wls
        n = interpolate.interp1d(grid, self.data[bin,:],'linear',bounds_error=False)
        return n(wls)

    def writeto(self, file, clobber=False,output_verify='exception'):
            hdu = self.hdu
            hdu.data = self.data
            hdu.header.update('CDELT1',self.step)
            hdu.header.update('CRVAL1',self.start)
            hdu.header.update('CRPIX1',1)
            hdu.writeto(file, clobber=clobber,output_verify=output_verify)
            self.filename = file

        
    def rebin(self, start=0.0, step=0.0, stop=0.0, zshift=0.0):
        """
        performs linear to linear rebinning and optional redshifting of the spectrum.
        rebinning is done through spline interpolation
        start is the starting wavelength of the original! spectrum
        stop  is the stop wavelength of the original! spectrum
        step  is the target step size 
        zshift (=0) target redshift.
        """
        import srebin
        if len( self.data.shape ) == 1:
            fluxv = self.data
            wl = self.grid()
            rwl, rf = srebin.linlinSpl(wl, fluxv, start=start, step=step, stop=stop, zshift=zshift)
            return Spectrum(rwl[0],rwl[1]-rwl[0],rf)    
        elif len( self.data.shape ) == 2:
            newdata = []
            rwl = 0
            for i in range(self.data.shape[1]):
                fluxv = self.data[i,:]
                wl = self.grid()
                rwl, rf = srebin.linlinspl(wl, fluxv, start=start, step=step, stop=stop, zshift=zshift)
                newdata.append(rf)
            return spectrum(rwl[0],rwl[1]-rwl[0],array(newdata) )    
        else:
            print( "ERROR: Can only deal with one or two dimensional arrays." )
            sys.exit(0)

    
def readSpectrum(file, extension=0, normalization=False):
    #hdulist = fits.open(file, ignore_missing_end=True)

    hdulist = fits.open(file)
    data = hdulist[extension].data
    hdulist.close()
    
    ndim = len( data.shape )
    if not ndim in [1,2,3]:
            print( "ERROR: Don't know how to deal with %d dim data." % ndim )
            sys.exit(1)
    if ndim == 3: # 3D cube
        step  = hdulist[extension].header['CDELT3']
        start = hdulist[extension].header['CRVAL3']
    else:
        step  = hdulist[extension].header['CDELT1']
        start = hdulist[extension].header['CRVAL1']

    try:
        if len(data.shape) == 3:
            crpix = hdulist[extension].header['CRPIX3']
        else:
            crpix = hdulist[extension].header['CRPIX1']
    except:
        crpix = 1
    #hdulist.close()

    if normalization:
        # normalize data by dividing through the exposure time
        data = data/float(hdulist[extension].header['exptime'])

    s  = Spectrum(start-(crpix-1)*step, step, data, filename=file)
    s.hdu = hdulist[0]
    return s
