import numpy as np
def colorCalc(flux, lam, redShift):
    rest_wl = lam/(1+redShift)
    #-----------
    def trap(x, y): return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))/2.
    #-----------
    nfil = 3
    fil = {}
    for q in range(nfil): fil[q] = np.loadtxt('../Summer2024/dSetton/psFilter'+str(q+1)+'.txt', skiprows=1)
    #-----------
    flux_nu = flux*rest_wl**2/(2.9979252458e18)
    #-----------
    integ = np.zeros(nfil) - 99.
    for q in range(nfil):
        integ[q] = (trap(fil[q][:,1], (np.interp(fil[q][:,1], rest_wl, flux_nu)* fil[q][:,2] / fil[q][:,1]))) / (trap(fil[q][:,1], (fil[q][:,2] / fil[q][:,1])))
    color = np.array([-2.5*np.log10(integ[0] / integ[1]), -2.5*np.log10(integ[1] / integ[2])])
    uMb, bMv = color[0], color[1]
    return uMb, bMv
def trap(x, y): return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))/2.
def hDeltaA(flux0, lam, redShift):
    #hdA = [4041.60,4079.75, 4083.50, 4122.25, 4128.50, 4161]
    hdA = [4017, 4057, 4083.50, 4122.25, 4153, 4193]
    #hdA = [ha*(1+redShift) for ha in hdA]
    lam = lam/(1+redShift)
    fL = np.mean(flux0[(lam>hdA[0])&(lam<hdA[1])])
    fR = np.mean(flux0[(lam>hdA[4])&(lam<hdA[5])])
    lL, lR = np.mean([hdA[0], hdA[1]]), np.mean([hdA[4], hdA[5]])
    coefficients = np.polyfit([lL, lR], [fL, fR], 1)
    #---------------------------------
    poly = np.poly1d(coefficients)
    lM = np.linspace(lL, lR, 100)
    fM = poly(lM)
    #---------------------------------
    ll = (lam>=hdA[2])&(lam<=hdA[3])
    lM = lam[ll]
    fC, fS = poly(lM), flux0[ll]
    int0 = trap(lM, 1-fS/fC)
    return int0
def dn4000_wren_fnu(lam, flam, flux_ivar, redShift):
    restLam = lam/(1+redShift)
    loglam = np.log10(restLam)
    # loglam = log10 rest-frame wavelength
    # flam = Flux in flambda
    # ivar = inverse variance of the flux
    # Defining the bandpasses
    bluelim = [np.log10(3850.),np.log10(3950.)]
    redlim = [np.log10(4000.),np.log10(4100)]
    #convert to fnu
    fnu = flam * (10**(loglam))**2
    ivar_nu = (flam/fnu)**2 * flux_ivar
    #--------------
    iblue = (loglam > bluelim[0]) & (loglam < bluelim[1])
    sweight = np.sum(ivar_nu[iblue])
    sflux = np.sum(fnu[iblue]*ivar_nu[iblue])
    bluec = sflux/sweight
    sig2bluec = 1./sweight
    ired = (loglam > redlim[0]) & (loglam < redlim[1])
    sweight = np.sum(ivar_nu[ired])
    sflux = np.sum(fnu[ired]*ivar_nu[ired])
    redc = sflux/sweight
    sig2redc = 1./sweight
    d4000n = redc/bluec
    ivar = (1./(d4000n**2))/(sig2redc/(redc**2) + sig2bluec/(bluec**2))
    return d4000n.value, 1./np.sqrt(ivar)