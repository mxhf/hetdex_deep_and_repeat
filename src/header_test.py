from astropy.io.fits import Header
from astropy.wcs import wcs
hh = Header()

hh["NAXIS"]   =                    3 
#hh["WCSAXES"]   =                    2
hh["NAXIS1"]  =                  142                                                  
hh["NAXIS2"]  =                  153                                                  
hh["NAXIS3"]  =                 1010 
hh["CTYPE3"] = 'Wave    '                                                            
hh["CRPIX3"] =                  1.0                                                  
hh["CRVAL3"] =              3494.74                                                  
hh["CDELT3"] =    1.985839800000122                                                  
hh["CTYPE1"] = 'RA---TAN'                                                            
hh["CRPIX1"] =    74.25405364037756                                                  
hh["CRVAL1"] =    149.9647523612416                                                  
hh["CDELT1"] = -0.00013888888888888                                                  
hh["CUNIT1"] = 'deg'                                                            
hh["CTYPE2"] = 'DEC--TAN'                                                            
hh["CRPIX2"] =    82.82783906313449                                                  
hh["CRVAL2"] =    2.140140717332689                                                  
hh["CDELT2"] = 0.000138888888888888                                                  
hh["CUNIT2"] = 'deg'                                                            
hh["CUNIT3"] = 'Angstrom'   


#hh["WCSAXES"] =                     2
#["CRPIX1"] =               -234.75
#["CRPIX2"] =                8.3393
#["CDELT1"] =             -0.066667
#["CDELT2"] =              0.066667
#["CUNIT1"] =  'deg'               
#["CUNIT2"] =  'deg'               
#["CTYPE1"] =  'RA---TAN'          
#["CTYPE2"] =  'DEC--TAN'          
#["CRVAL1"] =                   0.0
#["CRVAL2"] =                 -90.0
#hh["RADESYS"] =  'ICRS'              

wcs.WCS(hh)
#wcs.WCS(w.to_header())
