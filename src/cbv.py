import numpy as np
import pylab as pl
from VBLinRegARD import bayes_linear_fit_ard as VBF
from cdpp import cdpp
import astropy.io.fits as pyfits

def fit_basis(flux, basis, scl = None):
    '''
    weights = fit_basis(flux, basis, scl = None)
    fit VB linear basis model to one or more light curves

    Inputs:
        flux: (nobj x nobs) light curve(s) 
        basis: (nobs x nb) basis trends
        scl: (nb) prior scaling factors for the basis trends
    Outputs:
        weights: (nobj x nb) weights
    '''
    # pre-process basis
    nb,nobs = basis.shape
    B = np.matrix(basis.T)
    if scl == None: scl = np.ones(nb)
    Bnorm = np.multiply(B, scl)
    Bs = Bnorm.std()
    Bnorm /= Bs
    Bnorm = np.concatenate((Bnorm, np.ones((nobs,1))), axis=1)
    # array to store weights
    nobj = flux.shape[0]
    weights = np.zeros((nobj,nb))
    for iobj in np.arange(nobj): 
        # pre-process flux
        F = np.matrix(flux[iobj,:]).T
        l = np.isfinite(F)
        Fm = F.mean()
        Fs = F.std()
        Fnorm = (F - Fm) / Fs
        res = VBF(Bnorm, Fnorm)
        w, V, invV, logdetV, an, bn, E_a, L = res
        weights[iobj,:] = np.array(res[0][:-1]).flatten() * scl * Fs / Bs
    return weights

def apply_basis(weights, basis):
    '''
    model = apply_basis(weights, basis) 
    Compute linear basis model given weights and basis matrix

    Inputs:
        weights: (nobj x nb) weights
        basis: (nobs x nb) basis trends
    Outputs:
        corr: (nobj x nobs) correction to apply to light curves
    '''
    return np.dot(weights, basis)

def correct_flux(flux, cbv, nB = 4, use = None, doPlot = True):
    '''
    corrected_flux = correct_flux(flux, basis, nB = 4, use = None, \
                                  doPlot = True)
    Correct light curve for systematics by first nB CBVs using VB.

    Inputs:
        flux: (1-D array) light curves 
        cbv: (2-D array) co-trending basis vectors trends
    Optional inputs:
        nB: number of CBVs to use (the first nB are used)
        use: boolean array, True for data points to use in evaluating correction, 
             False for data points to ignore (NaNs are also ignored)
        doPlot: set to False to suppress plot
    Outputs:
        corrected_flux: (same shape as flux) corrected light curves
        weights: (nB array) basis vector coefficients
    '''
    nobs = len(flux)    
    if cbv.shape[1] == nobs: cbv_ = cbv[:nB,:]
    else: cbv_ = cbv[:,:nB].T
    corrected_flux = np.copy(flux)
    l = np.isfinite(flux)
    if use != None: l *= use
    weights = fit_basis(flux[l].reshape((1,l.sum())), cbv_[:,l])
    corr = apply_basis(weights, cbv_).reshape(flux.shape)
    corrected_flux = flux - corr
    if doPlot == True:
        pl.clf()
        x = np.arange(nobs)
        pl.plot(x, flux, '-', c = 'grey')
        pl.plot(x[l], flux[l], 'k-')
        pl.plot(x, corr, 'c-')
        pl.plot(x, corrected_flux, 'm-')
        pl.xlabel('Observation number')
        pl.xlabel('Flux')
    return corrected_flux, weights

def correct_file(infile, cbvfile, outfile, input_type = 'SAP', \
                 exclude_func = None, exclude_func_par = None, doplot = False):
    '''
    time, cadence, corrected_flux = correct_file_nB(infile, cbvfile, outfile, \
        input_type = 'SAP', exclude_func = None, \
        exclude_func_par = None)

    Correct light curve containined in infile using CBVs contained in
    cbvfile, using up to nBmax CBVs

    Inputs:
        infile: input (FITS) light curve file
        cbvfile: input (FITS) CBV file
        outfile: output (FITS) file to save results in. This is a copy of the input
            file with extra columns 'CBVX_FLUX' where X = 1, 2, ..., 8 containing the
            systematics corrected fluxes using 1, 2, ..., 8 CBVs, respectively. The
            weights associated with each CBV are saved in the header (CBVW_XY, where
            X is the number of CBVs used and Y the index of the CBV to which the
            weight is applied), and the 6.5-hour CDPPs after correction are also
            stored in the header (CDPP_CSX, where X is the number of CBVs used)
    Optional inputs:
        input_type: type of data to use as input. Options are:
            SAP: "raw" (simple aperture photometry) data
            JCR: "jump-corrected" data 
        exclude_func: function f(t,par), which returns list of indices to ignore
        exclude_func_par: parameters of exclude function
        doplot: if True, produce plots on screen
    '''
    nBmax = 8
    # Read in light curve data
    h1 = pyfits.open(infile, mode = 'readonly')
    kic = h1[0].header['KEPLERID']
    quarter = h1[0].header['QUARTER']
    module = h1[0].header['MODULE']
    output = h1[0].header['OUTPUT']
    print 'Reading in quarter %d light curve data for KIC %d.' % (quarter, kic)
    print 'Object is located on module %d, output channel %d.' \
      % (module, output)
    if input_type == 'SAP':
        print 'Reading SAP data'
        flux = h1[1].data.field('SAP_FLUX').astype('float64')
    elif input_type == 'JCR':
        print 'Reading JCR data'
        flux = h1[1].data.field('JCR_FLUX').astype('float64')
    else:
        print 'Error: input type %s not supported'
        return
    if doplot == True:
        pl.clf()
        time = h1[1].data.field('TIME').astype('float64')
        pdc = h1[1].data.field('PDCSAP_FLUX').astype('float64')
        l = np.isfinite(time)
        tmin = time[l].min()
        tmax = time[l].max()
    nobs = len(flux)
    l = np.isfinite(flux)
    nval = l.sum()
    print 'Read in %d observations of which %d valid.' % (nobs, nval)
    # Normalise
    # Read in CBV data
    cbv = np.zeros((nobs, 16))
    h2 = pyfits.open(cbvfile)
    if h2[0].header['QUARTER'] != quarter:
        print 'Error: CBV file is for quarter %d.' % h2[0].header['QUARTER']
        return
    n_ext = len(h2) - 1
    for i in np.arange(n_ext)+1:
        if h2[i].header['MODULE'] != module: continue
        if h2[i].header['OUTPUT'] != output: continue
        for j in np.arange(16):
            cbv[:,j] = h2[i].data.field('VECTOR_%d' % (j+1)).astype('float64')
        break
    h2.close()
    # Perform correction
    if exclude_func != None:
        if exclude_func_par == None:
            exclude_indices = exclude_func(time)
        else:
            exclude_indices = exclude_func(time, exclude_func_par)
        use = np.ones(nobs, 'bool')
        use[exclude_indices] = False
    else:
        use = None
    unit = h1[1].header['TUNIT4']
    cols = h1[1].columns
    if doplot == True:
        mms = np.median(flux[np.isfinite(flux)])
        mmp = np.median(pdc[np.isfinite(pdc)])
        pdc = pdc - mmp + mms
        sap_cdpp = cdpp(time, flux)
        pdc_cdpp = cdpp(time, pdc)
        print 'Input CDPP: %f' % sap_cdpp 
        print 'PDC CDPP: %f' % pdc_cdpp
        ax1 = pl.subplot(211)
        diff1 = flux[1:] - flux[:-1]
        ll1 = np.isfinite(diff1)
        mm1 = np.median(diff1[ll1])
        offset1 = 5 * 1.48 * np.median(abs(diff1 - mm1))
        pl.plot(time, flux, 'k-')
        pl.plot(time, flux - pdc + mms - offset1, 'g-')
        pl.ylabel('raw flux')
        pl.title('KID%d Q%d (module %d output %d)' % (kic, quarter, module, output))
        ax2 = pl.subplot(212, sharex = ax1)
        pl.plot(time, pdc, 'g-')        
        diff2 = pdc[1:] - pdc[:-1]
        ll2 = np.isfinite(diff2)
        mm2 = np.median(diff2[ll2])
        offset2 = 5 * 1.48 * np.median(abs(diff2 - mm2))
        pl.ylabel('corr. flux')
        pl.xlabel('time')
    for i in np.arange(nBmax):
        flux_cbv, weights = correct_flux(flux, cbv, nB = i + 1, use = use, \
                                         doPlot = False)
        for j in range(i+1):
            h1[1].header['CBVW_%d%d' % (i+1,j)] = repr(weights[0][j])
        mmc = np.median(flux_cbv[np.isfinite(flux_cbv)])
        flux_cbv = flux_cbv - mmc + mms
        cbv_cdpp = cdpp(time, flux_cbv)
        h1[1].header['CDPP_CS%d' % (i+1)] = repr(cdpp)
        if doplot == True:
            print 'CDPP with %d CBVs: %f' % (i+1, cbv_cdpp)
            print 'Weights:', weights
            corr = flux - flux_cbv + mms
            dr = i/float(nBmax-1)
            rgb = (1-dr,0,dr)
            pl.sca(ax1)
            pl.plot(time, corr - offset1 * (i+2), c = rgb)
            pl.sca(ax2)
            pl.plot(time, flux_cbv - offset2 * (i+1), c = rgb)
        col = pyfits.Column(name = 'CBV%d_FLUX' % (i + 1), format = 'E', \
                            disp = 'E14.7', unit = unit, \
                            array = flux_cbv)
        cols += col
    if doplot == True:
        pl.xlim(tmin, tmax)
    # Save
    hdr_save = h1
    h1[1] = pyfits.BinTableHDU.from_columns(cols, header=h1[1].header)
    print 'Saving to file %s' % outfile
    h1.writeto(outfile, clobber = True)
    h1.close()
    return 