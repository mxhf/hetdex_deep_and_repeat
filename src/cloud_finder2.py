import time
import numpy as np
import spectrum
from collections import OrderedDict
import argparse
from astropy.io import fits
from astropy import wcs
import sys
from astropy.io import ascii



def pp(s):
    print(s)
    return s + "\n"


def grow_segment(c, outmap, x, y, z, threshold, label):
    # Non object oriented version
    def get_children(zyx, shape):
        maxz, maxy, maxx = shape
        z, y, x = zyx
        children = []
        ddx = [-1,0,1]
        ddy = [-1,0,1]
        ddz = [-1,0,1]
        #ddz = [0]
        for dx in ddx:
            for dy in ddy:
                for dz in ddz:
                    if not dx == dy == dz == 0:
                        newx, newy, newz = x+dx, y+dy, z+dz
                        if newx >= 0 and newx < maxx\
                            and newy >= 0 and newy < maxy\
                            and newz >= 0 and newz < maxz:
                               children.append( (newz, newy, newx) ) 
        return children

    pixel_list = []
    stack = [(z,y,x)]
    outmap[(z,y,x)] = label

    maxiter = 1e6
    iter = 0
    while len(stack) > 0: 
        pm = stack.pop()
        pp = get_children(pm, c.shape)
        for p in pp:
            if c[p] >= threshold and  outmap[p] == 0: # pixel that are labeled 0 have not been visited yet
                stack.append(p)
                outmap[p] = label
            elif c[p] < threshold and  outmap[p] == 0:
                outmap[p] = -1 # label pixel with -1 if they have beed visited already
        iter += 1
        if iter > maxiter:
            break

    return outmap#pixel_list            


def build_map(c, detect_threshold):
    # run cloud growth algorithm on all pixels above detection threshold
    # try to be intelligent, only loop over pixels that exceed detection threshold
    ii = c > detect_threshold 
    N =  (np.sum(ii))
    print("{} pixel above detection threshold".format(N))

    outmap = np.zeros_like(c, dtype=int)

    zz,yy,xx = [np.arange(s, dtype=int) for s in c.shape]
    YY,ZZ,XX = np.meshgrid(yy, zz, xx)

    label = 1
    for i, (x,y,z) in enumerate( zip(XX[ii], YY[ii], ZZ[ii] ) )  :  
        if outmap[z,y,x] == 0:
            pp("### {} z={} ###".format(i,z))
            print(x,y,z, outmap[z,y,x])
            #outmap = build_map2(c, outmap, x,y,z, threshold = grow_threshold, label=label)

            summary = ""
            summary += pp("Building map starting with pixel {} out of {} that exceeds threshold...".format(i,N))
            start_time = time.time()
            #print("{} labeled pixels on map".format( np.nansum( (outmap > 0).flatten())) )
            outmap = grow_segment(c, outmap, x,y,z, threshold = grow_threshold, label=label)
            time_to_build = time.time() - start_time
            summary += pp("Time to build map: {:.4e} s".format(time_to_build))

            print("{} labeled pixels on map".format( np.nansum( (outmap > 0).flatten())) )
            print("{} untouched pixels on map".format( np.nansum( (outmap == 0).flatten())) )

        label += 1

        if i > 1e6:
            break
    #print (cnt)
    return outmap


def filter_minsize(outmap, minsize):
    # filter regions with too small volume 
    rr = np.sort( np.unique( outmap.flatten() ) )
    for r in rr:
        if not r > 0:
            continue
        N = np.sum( outmap == r )
        #print("{} : N = {}".format(r, N))
        if N < minsize:
            outmap[outmap == r] = -1  
    rr = np.sort( np.unique( outmap.flatten() ) )
    
    # relabel
    for i,r in enumerate(rr[rr>0]):
        outmap[outmap == r] = i + 1

    rr = np.sort( np.unique( outmap.flatten() ) )
    print("{} regions survive size cut".format( len(rr[rr>0]) ))

    return outmap



def save_map(outmap, fmapout):
    w = wcs.WCS(s.hdu)
    # save map
    f = np.zeros_like(c)
    f[outmap > 0] = c[outmap > 0]
    wcs_header =  w.to_header()


    h = fits.PrimaryHDU(data=outmap, header=s.hdu.header)
    for k in wcs_header:
        h.header[k] = wcs_header[k]
    hdu = fits.HDUList(h)

    # save map filtered data
    f = np.zeros_like(c)
    f[outmap > 0] = c[outmap > 0]
    h = fits.ImageHDU(data=f, header=s.hdu.header, name = "filtered_data")
    for k in wcs_header:
        h.header[k] = wcs_header[k]

    hdu.append(h)

    # save shells 
    f = np.zeros_like(c)
    f[outmap == -1] = c[outmap == -1]

    h = fits.ImageHDU(data=f, header=s.hdu.header, name = "shells")
    for k in wcs_header:
        h.header[k] = wcs_header[k]
    hdu.append(h)

    hdu.writeto(fmapout, overwrite=True)
    print("Wrote {}.".format(fmapout))



parser = argparse.ArgumentParser()
parser.add_argument("-m", "--minsize", type=int, default=3,
                    help="Minimum region size.")
parser.add_argument("-g", "--grow_threshold", type=float,  default=0.305,
                    help="Region growth threshold.")
parser.add_argument("-d","--detect_threshold", type=float, default=0.6,
                    help="Detection threshold.")
parser.add_argument('-i', '--infile', type=str,                    
                    help='Input cube.')
parser.add_argument('-o', '--outfile', type=str, default='',
                    help='Output map.')
parser.add_argument('-b', '--bad_regions', type=str, default='',
                    help='Output map.')
args = parser.parse_args(sys.argv[1:])

badwlregions = []

t = ascii.read(args.bad_regions)
for r in t:
    badwlregions.append([r["start"],r["end"]])



fcube = args.infile
fmapout = args.outfile
detect_threshold = args.detect_threshold
grow_threshold = args.grow_threshold
minsize = args.minsize


s = spectrum.readSpectrum(fcube)
c = s.data
for r in badwlregions:  
    c[r[0]:r[1]] = 0. # take out bad region
    
outmap = build_map(c, detect_threshold)
outmap = filter_minsize(outmap, minsize)


save_map(outmap, fmapout)
