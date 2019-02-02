#!/usr/bin/env python
import sys
from astropy.io import fits

infile = sys.argv[1]
lcutoff = float( sys.argv[2] )
hcutoff = float( sys.argv[3] )
outfile = sys.argv[4]

hdu = fits.open(infile)

hdu[0].header["CTYPE1"] = "RA---TAN"
hdu[0].header["CUNIT3"] = "Angstrom"
hdu[0].data[ hdu[0].data < lcutoff ] = lcutoff
hdu[0].data[ hdu[0].data > hcutoff ] = hcutoff
hdu.writeto(outfile, overwrite=True)
