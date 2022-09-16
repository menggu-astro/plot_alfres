### ---------------- #
### 1. MCMC corner plot, for 3 ALF modes. done
### 1.1. 3 modes posterior dist, done
### 1.2. trace plot Alexa's code
### 1.3. corner plot, done
### 2. Spectral comparison, for 3 subplots, and 5 subplots
###    5 subplots done
### Meng Gu, Jan 4th, 2017
### ---------------- #


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from astropy.io import ascii
import warnings
import math
import scipy; from scipy import constants
from scipy import integrate, constants, interpolate

# ----------------------------------------------------------------#
class ALFbest(object):
    """
    example:
        dir_input = '/.../alfinput_Nov8'
        dir_output = '/.../alfresult_Dec1'
        fname = 'med_NGC4874'
        appdx = '1'
        test = ALFbest(respath = dir_output, inpath = dir_input, fname = fname, appdx=appdx)
        print(test.mean['FeH'], test.mean['zH'])
        test.libcorrect()
        print(test.mean['FeH'], test.mean['aH'], test.mean['aFe'])

    """
    def __init__(self, respath, inpath, fname, appdx, outdir, alfmode = 0,):
        """
        respath: path to alfresults
        inpath: path to alfinput
        fname: spectra name
        outdir: output dir for figure, etc
        """
        self.mode = alfmode
        self.name = fname
        self.appdx = appdx
        if appdx is None:
            self.fullname = fname
        else:
            self.fullname = fname+'_'+appdx
        self.inpath = '{0}/{1}'.format(inpath, self.name)
        self.respath = '{0}/{1}'.format(respath, self.fullname)

        self.outdir = outdir
        tablelength = len(ascii.read('{0}.sum'.format(self.respath)).columns)
        #print('Outputs containt', tablelength, 'columns.\n')
        self.bestspec   = ascii.read('{0}.bestspec'.format(self.respath))
        try:
            self.inspec = ascii.read('{0}.dat'.format(self.inpath))
        except:
            warning = ('Do not have the input data file')
            warnings.warn(warning)
            self.inspec = None


        if tablelength == 52:
            self.labels = ['chi2','velz','sigma','logage','zH',
                  'FeH', 'aH', 'CH', 'NH', 'NaH', 'MgH',
                  'SiH', 'KH', 'CaH', 'TiH','VH', 'CrH',
                  'MnH', 'CoH', 'NiH', 'CuH', 'SrH','BaH',
                  'EuH', 'Teff', 'IMF1', 'IMF2', 'logfy',
                  'sigma2', 'velz2', 'logm7g', 'hotteff',
                  'loghot','fy_logage','logemline_H',
                  'logemline_Oiii','logemline_Sii', 'logemline_Ni',
                  'logemline_Nii','logtrans', 'jitter','logsky', 'IMF3', 'IMF4', 'h3','h4',
                  'ML_r','ML_i','ML_k','MW_r', 'MW_i','MW_k']
        elif tablelength == 53:
            self.labels = ['chi2','velz','sigma','logage','zH',
                           'FeH', 'aH', 'CH', 'NH', 'NaH', 'MgH',
                           'SiH', 'KH', 'CaH', 'TiH',
                           'VH', 'CrH','MnH', 'CoH', 'NiH', 'CuH', 'SrH','BaH', 'EuH',
                           'Teff', 'IMF1', 'IMF2', 'logfy',
                           'sigma2', 'velz2','logm7g', 'hotteff', 'loghot', 'fy_logage',
                           'logemline_H', 'logemline_Oii','logemline_Oiii', 'logemline_Sii',
                           'logemline_Ni', 'logemline_Nii','logtrans', 'jitter',
                           'logsky',
                           'IMF3', 'IMF4', 'h3', 'h4', 'ML_r','ML_i','ML_k','MW_r', 'MW_i','MW_k']

        elif tablelength == 50:
            self.labels = ['chi2','velz','sigma','logage','zH','FeH', 'aH', 'CH', 'NH', 'NaH',
                           'MgH','SiH', 'KH', 'CaH', 'TiH','VH', 'CrH','MnH', 'CoH', 'NiH', 'CuH',
                           'SrH','BaH', 'EuH','Teff', 'IMF1', 'IMF2', 'logfy','sigma2', 'velz2',
                           'logm7g', 'hotteff', 'loghot','fy_logage','logtrans','logemline_H',
                           'logemline_Oiii','logemline_Sii','logemline_Ni', 'logemline_Nii','jitter','IMF3',
                           'logsky','IMF4','ML_r','ML_i','ML_k','MW_r', 'MW_i','MW_k']

        elif tablelength == 55:
            self.labels = ['chi2','velz','sigma','logage','zH',
                           'FeH', 'aH', 'CH', 'NH', 'NaH', 'MgH',
                           'SiH', 'KH', 'CaH', 'TiH','VH', 'CrH',
                           'MnH', 'CoH', 'NiH', 'CuH', 'SrH','BaH', 'EuH',
                           'Teff', 'IMF1', 'IMF2', 'logfy',
                           'sigma2', 'velz2','logm7g', 'hotteff', 'loghot', 'fy_logage',
                           'logemline_H', 'logemline_Oii','logemline_Oiii', 'logemline_Sii',
                           'logemline_Ni', 'logemline_Nii','logtrans', 'jitter', 'logsky',
                           'IMF3', 'IMF4', 'h3', 'h4', 'velz3', 'logfrac3',
                           'ML_r','ML_i','ML_k','MW_r', 'MW_i','MW_k']

        elif tablelength == 56:
            self.labels = ['chi2','velz','sigma','logage','zH',
                           'FeH', 'aH', 'CH', 'NH', 'NaH', 'MgH',
                           'SiH', 'KH', 'CaH', 'TiH','VH', 'CrH',
                           'MnH', 'CoH', 'NiH', 'CuH', 'SrH','BaH', 'EuH',
                           'Teff', 'IMF1', 'IMF2', 'logfy',
                           'sigma2', 'velz2','logm7g', 'hotteff', 'loghot', 'fy_logage',
                           'logemline_H', 'logemline_Oii','logemline_Oiii', 'logemline_Sii',
                           'logemline_Ni', 'logemline_Nii','logtrans', 'jitter', 'logsky',
                           'IMF3', 'IMF4', 'h3', 'h4', 'velz3', 'logfrac3', 'sigma3',
                           'ML_r','ML_i','ML_k','MW_r', 'MW_i','MW_k']

        elif tablelength == 60:
            self.labels = ['chi2','velz','sigma','logage','zH',
                           'FeH', 'aH', 'CH', 'NH', 'NaH', 'MgH',
                           'SiH', 'KH', 'CaH', 'TiH','VH', 'CrH',
                           'MnH', 'CoH', 'NiH', 'CuH', 'SrH','BaH', 'EuH',
                           'Teff', 'IMF1', 'IMF2', 'logfy',
                           'sigma2', 'velz2','logm7g', 'hotteff', 'loghot', 'fy_logage',
                           'logemline_H', 'logemline_Oii','logemline_Oiii', 'logemline_Sii',
                           'logemline_Ni', 'logemline_Nii','logtrans', 'jitter', 'logsky',
                           'IMF3', 'IMF4', 'h3', 'h4',
                           'velz3', 'frac3', 'logage3', 'zH3', 'FeH3', 'aH3', 'MgH3',
                           'ML_r','ML_i','ML_k','MW_r', 'MW_i','MW_k']

        #print('tablelength=', tablelength)
        #print self.labels

        try:
            self.mcmc = ascii.read('{0}.mcmc'.format(self.respath), format='no_header',
                                   names=self.labels)#, fast_reader={'parallel': True})
        except:
            warning = ('Do not have the *.mcmc file')
            warnings.warn(warning)
            self.mcmc = None


        alfres = ascii.read('{0}.sum'.format(self.respath), names=self.labels)
        with open('{0}.sum'.format(self.respath)) as f:
                for line in f:
                    if line[0] == '#':
                        if 'Nwalkers' in line:  self.nwalkers = float(line.split('=')[1].strip())
                        elif 'Nchain' in line:  self.nchain = float(line.split('=')[1].strip())
                        elif 'Nsample' in line:  self.nsample = float(line.split('=')[1].strip())


        if len(self.labels) != len(alfres.colnames):
            error = ('Label array and parameter array have different lengths.')
            raise ValueError(error)

        """
        0:   Mean of the posterior
        1:   Parameter at chi^2 minimum
        2:   1 sigma error
        3-7: 2.5%, 16%, 50%, 84%, 97.5% CLs
        8-9: lower and upper priors
        """
        self.mean = dict(zip(self.labels, alfres[0]))
        self.minchi2 = dict(zip(self.labels, alfres[1]))
        self.onesigma = dict(zip(self.labels, alfres[2]))
        self.cl25 = dict(zip(self.labels, alfres[3]))
        self.cl16 = dict(zip(self.labels, alfres[4]))
        self.cl50 = dict(zip(self.labels, alfres[5]))
        self.cl84 = dict(zip(self.labels, alfres[6]))
        self.cl98 = dict(zip(self.labels, alfres[7]))
        self.lo_prior = dict(zip(self.labels, alfres[8]))
        self.up_prior = dict(zip(self.labels, alfres[9]))

        self.mcmc['velz'] = self.mcmc['velz'].astype(float)
        self.cl25['velz'] = self.cl25['velz'].astype(float)
        self.cl16['velz'] = self.cl16['velz'].astype(float)
        self.cl50['velz'] = self.cl50['velz'].astype(float)
        self.cl84['velz'] = self.cl84['velz'].astype(float)
        self.cl98['velz'] = self.cl98['velz'].astype(float)


    def libcorrect(self):
        """
        Need to correct the raw abundance values given by ALF.
        Use the metallicity-dependent correction factors from the literature.
        To-Do:
            Only correcting the mean of the posterior values for now.
            Correct other parameters later.
        """

        #;Schiavon 2007
        #libmgfe = [0.4,0.4,0.4,0.4,0.29,0.20,0.13,0.08,0.05,0.04]
        #libcafe = [0.32,0.3,0.28,0.26,0.20,0.12,0.06,0.02,0.0,0.0]
        # library correction factors from Schiavon 2007, Table 6;
        libfeh  = [-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2]
        libofe  = [0.6,0.5,0.5,0.4,0.3,0.2,0.2,0.1,0.0,0.0]
        #libmgfe = [0.4,0.4,0.4,0.4,0.29,0.20,0.13,0.08,0.0,0.0]
        #libcafe = [0.32,0.3,0.28,0.26,0.20,0.12,0.06,0.02,0.0,0.0]
        # Bensby et al. 2014
        #;fitted to Milone et al. 2011 HR MILES stars
        libmgfe = [0.4,0.4,0.4,0.4,0.34,0.22,0.14,0.11,0.05,0.04]
        #;from B14
        libcafe = [0.32,0.3,0.28,0.26,0.26,0.17,0.12,0.06,0.0,0.0]
        #libmgfe = [0.4,0.4,0.4,0.38,0.37,0.27,0.21,0.12,0.05,0.0]
        #libcafe = [0.32,0.3,0.28,0.26,0.26,0.17,0.12,0.06,0.0,0.0]


        # In ALF the oxygen abundance is used a proxy for alpha abundance
        #spl_delafe = interpolate.UnivariateSpline(libfeh, libofe, s=1, k=1)
        #spl_delmgfe = interpolate.UnivariateSpline(libfeh, libmgfe, s=1, k=1)
        #spl_delcafe = interpolate.UnivariateSpline(libfeh, libcafe, s=1, k=1)
        # why? --> NOTE: Forcing factors to be 0 for [Fe/H]=0.0,0.2
        spl_delafe = interpolate.interp1d(libfeh, libofe,kind='linear',bounds_error=False,fill_value='extrapolate')
        spl_delmgfe = interpolate.interp1d(libfeh, libmgfe, kind='linear',bounds_error=False,fill_value='extrapolate')
        spl_delcafe = interpolate.interp1d(libfeh, libcafe, kind='linear',bounds_error=False,fill_value='extrapolate')
        #del_alfe = interpolate.interp1d(lib_feh, lib_ofe, kind='linear', bounds_error=False, fill_value='extrapolate')
        #del_mgfe = interpolate.interp1d(lib_feh, lib_mgfe, kind='linear', bounds_error=False, fill_value='extrapolate')
        #del_cafe = interpolate.interp1d(lib_feh, lib_cafe, kind='linear', bounds_error=False, fill_value='extrapolate')

        # .mcmc file first
        al_mcmccorr = spl_delafe(np.copy(self.mcmc['zH']))
        mg_mcmccorr = spl_delmgfe(np.copy(self.mcmc['zH']))
        ca_mcmccorr = spl_delcafe(np.copy(self.mcmc['zH']))

        udlabel = ['aFe', 'MgFe', 'SiFe', 'CaFe', 'TiFe', 'CFe',
                   'NFe', 'NaFe', 'KFe', 'VFe', 'CrFe', 'MnFe',
                   'CoFe', 'NiFe', 'CuFe', 'SrFe', 'BaFe', 'EuFe']
        for ilabel in udlabel:
             xFe_mcmc = self.mcmc[ilabel[:-2]+'H'] - self.mcmc['FeH']
             self.mcmc[ilabel] = np.copy(xFe_mcmc)

        self.mcmc['aFe'] =  np.copy(self.mcmc['aFe']) + al_mcmccorr
        self.mcmc['MgFe'] =  np.copy(self.mcmc['MgFe']) + mg_mcmccorr
        self.mcmc['CaFe'] =  np.copy(self.mcmc['CaFe']) + ca_mcmccorr
        self.mcmc['SiFe'] =  np.copy(self.mcmc['SiFe']) + ca_mcmccorr
        self.mcmc['TiFe'] =  np.copy(self.mcmc['TiFe']) + ca_mcmccorr
        self.mcmc['FeH'] = self.mcmc['FeH'] + self.mcmc['zH']  # mcmc.Fe is updated


        # update mean, cl50, cl16, cl84, minchi2, onesigma for all elements
        old_minchi2FeH = float(self.minchi2['FeH'])
        old_onesigmaFeH = float(self.onesigma['FeH'])

        udlabel = ['aFe', 'MgFe', 'SiFe', 'CaFe', 'TiFe', 'CFe',
                   'NFe', 'NaFe', 'KFe', 'VFe', 'CrFe', 'MnFe',
                   'CoFe', 'NiFe', 'CuFe', 'SrFe', 'BaFe', 'EuFe']
        for ilabel in udlabel:
            self.mean.update({ilabel: float( np.nanmean(self.mcmc[ilabel]) )})  #lib corrected
            self.cl50.update({ilabel: float( np.percentile(self.mcmc[ilabel], 50))})  #lib corrected
            self.cl16.update({ilabel: float( np.percentile(self.mcmc[ilabel], 16))})  #lib corrected
            self.cl84.update({ilabel: float( np.percentile(self.mcmc[ilabel], 84))})  #lib corrected
            self.minchi2.update({ilabel: float( self.minchi2[ilabel[:-2]+'H'] ) - old_minchi2FeH}) # NOT corrected
            self.onesigma.update({ilabel: math.sqrt(float( self.onesigma[ilabel[:-2]+'H'] )**2. + old_onesigmaFeH**2.)})

        # update aFe, MgFe, CaFe, SiFe and TiFe with lib correction, only for .minchi2
        self.minchi2['aFe'] =  self.minchi2['aFe'] + spl_delafe(float(self.minchi2['zH']))
        self.minchi2['MgFe'] =  self.minchi2['MgFe'] + spl_delmgfe(float(self.minchi2['zH']))
        self.minchi2['CaFe'] =  self.minchi2['CaFe'] + spl_delcafe(float(self.minchi2['zH']))
        self.minchi2['SiFe'] =  self.minchi2['SiFe'] + spl_delcafe(float(self.minchi2['zH']))
        self.minchi2['TiFe'] =  self.minchi2['TiFe'] + spl_delcafe(float(self.minchi2['zH']))

        # ------ #
        # update FeH at last
        self.mean['FeH'] = float(np.nanmean(self.mcmc['FeH']))
        self.minchi2['FeH'] = float(self.minchi2['FeH']) + float(self.minchi2['zH'])
        self.cl50['FeH'] = float( np.percentile(self.mcmc['FeH'], 50) )
        self.cl16['FeH'] = float( np.percentile(self.mcmc['FeH'], 16) )
        self.cl84['FeH'] = float( np.percentile(self.mcmc['FeH'], 84) )
        self.onesigma['FeH'] = math.sqrt(float(self.onesigma['FeH'])**2. + float(self.onesigma['zH'])**2.)

        # update velocity dispersion
        self.mcmc['sigma'] = np.sqrt(np.copy(self.mcmc['sigma'])**2. + 100.**2.)
        self.mean['sigma'] = np.nanmean( self.mcmc['sigma'] )
        self.minchi2['sigma'] = np.sqrt(np.copy(self.minchi2['sigma'])**2. + 100.**2.)
        self.cl50['sigma'] = np.percentile( self.mcmc['sigma'], 50 )
        self.cl16['sigma'] = np.percentile( self.mcmc['sigma'], 16 )
        self.cl84['sigma'] = np.percentile( self.mcmc['sigma'], 84 )
        self.onesigma['sigma'] = np.nanstd( self.mcmc['sigma'] )


        # add total metallicity
        self.mcmc['totZH'] = self.mcmc['FeH'] + 0.94*self.mcmc['MgFe']
        self.mean['totZH'] = float(np.nanmean(self.mcmc['totZH']))
        self.minchi2['totZH'] = float(self.minchi2['FeH']) + 0.94*float(self.minchi2['MgFe'])
        self.cl50['totZH'] = float( np.percentile(self.mcmc['totZH'], 50) )
        self.cl16['totZH'] = float( np.percentile(self.mcmc['totZH'], 16) )
        self.cl84['totZH'] = float( np.percentile(self.mcmc['totZH'], 84) )
        self.onesigma['totZH'] = np.nanstd(self.mcmc['totZH'])


        if 'FeH3' in self.labels:
            al_mcmccorr3 = spl_delafe(np.copy(self.mcmc['zH3']))
            mg_mcmccorr3 = spl_delmgfe(np.copy(self.mcmc['zH3']))
            udlabel = ['aFe3', 'MgFe3', ]
            for ilabel in udlabel:
                xFe_mcmc = self.mcmc[ilabel[:-3]+'H3'] - self.mcmc['FeH3']
                self.mcmc[ilabel] = np.copy(xFe_mcmc)

            self.mcmc['aFe3'] =  np.copy(self.mcmc['aFe3']) + al_mcmccorr3
            self.mcmc['MgFe3'] =  np.copy(self.mcmc['MgFe3']) + mg_mcmccorr3
            self.mcmc['FeH3'] = self.mcmc['FeH3'] + self.mcmc['zH3']  # mcmc.Fe is updated

            old_minchi2FeH = float(self.minchi2['FeH3'])
            old_onesigmaFeH = float(self.onesigma['FeH3'])
            udlabel = ['aFe3', 'MgFe3', ]
            for ilabel in udlabel:
                self.mean.update({ilabel: float( np.nanmean(self.mcmc[ilabel]) )})  #lib corrected
                self.cl50.update({ilabel: float( np.percentile(self.mcmc[ilabel], 50))})  #lib corrected
                self.cl16.update({ilabel: float( np.percentile(self.mcmc[ilabel], 16))})  #lib corrected
                self.cl84.update({ilabel: float( np.percentile(self.mcmc[ilabel], 84))})  #lib corrected
                self.minchi2.update({ilabel: float( self.minchi2[ilabel[:-3]+'H3'] ) - old_minchi2FeH}) # NOT corrected
                self.onesigma.update({ilabel: math.sqrt(float( self.onesigma[ilabel[:-3]+'H3'] )**2. + old_onesigmaFeH**2.)})

            self.minchi2['aFe3'] =  self.minchi2['aFe3'] + spl_delafe(float(self.minchi2['zH3']))
            self.minchi2['MgFe3'] =  self.minchi2['MgFe3'] + spl_delmgfe(float(self.minchi2['zH3']))
            # update FeH at last
            self.mean['FeH3'] = float(np.nanmean(self.mcmc['FeH3']))
            self.minchi2['FeH3'] = float(self.minchi2['FeH3']) + float(self.minchi2['zH3'])
            self.cl50['FeH3'] = float( np.percentile(self.mcmc['FeH3'], 50) )
            self.cl16['FeH3'] = float( np.percentile(self.mcmc['FeH3'], 16) )
            self.cl84['FeH3'] = float( np.percentile(self.mcmc['FeH3'], 84) )
            self.onesigma['FeH3'] = math.sqrt(float(self.onesigma['FeH3'])**2. + float(self.onesigma['zH3'])**2.)

            # update velocity dispersion
            self.mcmc['sigma3'] = np.sqrt(np.copy(self.mcmc['sigma3'])**2. + 100.**2.)
            self.mean['sigma3'] = np.nanmean( self.mcmc['sigma3'] )
            self.minchi2['sigma3'] = np.sqrt(np.copy(self.minchi2['sigma3'])**2. + 100.**2.)
            self.cl50['sigma3'] = np.percentile( self.mcmc['sigma3'], 50 )
            self.cl16['sigma3'] = np.percentile( self.mcmc['sigma3'], 16 )
            self.cl84['sigma3'] = np.percentile( self.mcmc['sigma3'], 84 )
            self.onesigma['sigma3'] = np.nanstd( self.mcmc['sigma3'] )




    def imfdist(self):
        """need to load .mcmc file"""
        if self.mcmc is None:
            self.mcmc = ascii.read('{0}.mcmc'.format(self.respath), names=self.labels)
        self.mcmc['MLMW_r'] = self.mcmc['ML_r']/self.mcmc['MW_r']
        self.mcmc['MLMW_i'] = self.mcmc['ML_i']/self.mcmc['MW_i']

        temlist = self.mcmc['ML_r'][np.where(self.mcmc['chi2']==np.nanmin(self.mcmc['chi2']))]
        self.minchi2['ML_r'] = np.nanmean(temlist)
        #if np.nanstd(temlist)> 1e-2: print 'std M/L in r', np.nanstd(temlist)
        temlist = self.mcmc['ML_i'][np.where(self.mcmc['chi2']==np.nanmin(self.mcmc['chi2']))]
        self.minchi2['ML_i'] = np.nanmean(temlist)
        #if np.nanstd(temlist)> 1e-2: print 'std M/L in i', np.nanstd(temlist)
        temlist = self.mcmc['MLMW_r'][np.where(self.mcmc['chi2']==np.nanmin(self.mcmc['chi2']))]
        self.minchi2['MLMW_r'] = np.nanmean(temlist)
        #if np.nanstd(temlist)> 1e-2: print 'std M/L/MLMW in r', np.nanstd(temlist)
        temlist = self.mcmc['MLMW_i'][np.where(self.mcmc['chi2']==np.nanmin(self.mcmc['chi2']))]
        self.minchi2['MLMW_i'] = np.nanmean(temlist)
        #if np.nanstd(temlist)> 1e-2: print 'std M/L/MLMW in i', np.nanstd(temlist)

        self.mean['MLMW_r'] =  np.nanmean(self.mcmc['MLMW_r'])
        self.onesigma['MLMW_r'] =  np.nanstd(self.mcmc['MLMW_r'])
        self.cl25['MLMW_r'] = np.percentile(self.mcmc['MLMW_r'], 2.5)
        self.cl16['MLMW_r'] = np.percentile(self.mcmc['MLMW_r'], 16)
        self.cl50['MLMW_r'] = np.percentile(self.mcmc['MLMW_r'], 50)
        self.cl84['MLMW_r'] = np.percentile(self.mcmc['MLMW_r'], 84)
        self.cl98['MLMW_r'] = np.percentile(self.mcmc['MLMW_r'], 98)

        self.mean['MLMW_i '] =  np.nanmean(self.mcmc['MLMW_i'])
        self.onesigma['MLMW_i '] =  np.nanstd(self.mcmc['MLMW_i'])
        self.cl25['MLMW_i'] = np.percentile(self.mcmc['MLMW_i'], 2.5)
        self.cl16['MLMW_i'] = np.percentile(self.mcmc['MLMW_i'], 16)
        self.cl50['MLMW_i'] = np.percentile(self.mcmc['MLMW_i'], 50)
        self.cl84['MLMW_i'] = np.percentile(self.mcmc['MLMW_i'], 84)
        self.cl98['MLMW_i'] = np.percentile(self.mcmc['MLMW_i'], 98)

        self.cl16['chi2'] = np.nanpercentile(self.mcmc['chi2'], 16)
        self.cl50['chi2'] = np.nanpercentile(self.mcmc['chi2'], 50)
        self.cl84['chi2'] = np.nanpercentile(self.mcmc['chi2'], 84)




    def plot_trace(self):
        from matplotlib.backends.backend_pdf import PdfPages
        self.txtmcmc = np.loadtxt('{0}.mcmc'.format(self.respath))

        with open('{0}.sum'.format(self.respath), 'r') as temfile:
            temlines = temfile.readlines()
        for il, line in enumerate(temlines):
            if 'Nwalkers' in line.split():  self.nwalkers = int(line.split()[-1])
            if 'Nchain' in line.split():  self.nchain = int(line.split()[-1])

        num = len(self.labels)
        data = np.zeros((self.nchain, self.nwalkers, num))
        for i in range(0, self.nchain):
            for j in range(0, self.nwalkers):
                data[i,j] = self.txtmcmc[i*self.nwalkers + j]

        outname = (self.outdir +'/trace_{0}.pdf'.format(self.fullname))
        with PdfPages(outname) as pdf:
            for i, (label, trace) in enumerate(zip(self.labels, data.T)):
                fig = plt.figure(figsize=(8,6), facecolor='white')
                if i == 0: # Don't care to see the chi^2 value
                    continue
                plt.plot(np.arange(0, self.nchain),
                         data[:,:,i], color='k', alpha=0.1)
                plt.axhline(self.mean[label], color='#3399ff')
                plt.xlabel('Step')
                plt.ylabel(label)
                pdf.savefig()
                plt.close()
                plt.cla()

        return 0



    def alfpanel(self):
        fig = plt.figure(figsize=(16, 9),facecolor='white')
        ax1 = fig.add_axes([0.1, 0.7, 0.28, 0.28])
        ax2 = fig.add_axes([0.4, 0.7, 0.28, 0.28]);
        ax3 = fig.add_axes([0.7, 0.7, 0.28, 0.28])
        ax4 = fig.add_axes([0.1, 0.21, 0.43, 0.28]);
        ax5 = fig.add_axes([0.55, 0.21, 0.43, 0.28])

        ax1b = fig.add_axes([0.1, 0.55, 0.28, 0.13]);  ax2b = fig.add_axes([0.4, 0.55, 0.28, 0.13]);  ax3b = fig.add_axes([0.7, 0.55, 0.28, 0.13])
        ax4b = fig.add_axes([0.1, 0.06, 0.43, 0.13]);  ax5b = fig.add_axes([0.55, 0.06, 0.43, 0.13])

        ax1.set_xlim(3800, 4700);  ax2.set_xlim(4700, 5700);  ax3.set_xlim(5700, 6700)
        ax4.set_xlim(8000, 8920);  ax5.set_xlim(9630, 10100);
        ax1b.set_xlim(3800, 4700);  ax2b.set_xlim(4700, 5700);  ax3b.set_xlim(5700, 6700);
        ax4b.set_xlim(8000, 8920);  ax5b.set_xlim(9630, 10100)
        for ax in [ax1, ax4]: ax.set_ylabel('Flux', fontsize=22)
        for ax in [ax1b, ax4b]:
            ax.axhline(0, color='y', ls='-', alpha=0.35, lw=1)
            ax.set_ylabel('Residual/Flux', fontsize=18)
        for ax in [ax4b, ax5b]: ax.set_xlabel('wavelength', fontsize=20)

        return fig, ax1, ax2, ax3, ax4, ax5, ax1b, ax2b, ax3b, ax4b, ax5b



    def alfpanel2(self):
        fig = plt.figure(figsize=(16, 9),facecolor='white')
        ax1 = fig.add_axes([0.08, 0.70, 0.88, 0.28])
        ax1b = fig.add_axes([0.08, 0.55, 0.88, 0.13])
        ax2 = fig.add_axes([0.08, 0.21, 0.88, 0.28])
        ax2b = fig.add_axes([0.08, 0.06, 0.88, 0.13])

        ax1.set_xlim(3800, 4700);  ax2.set_xlim(4700, 5700)
        ax1b.set_xlim(3800, 4700);  ax2b.set_xlim(4700, 5700)
        for ax in [ax1, ax2]: ax.set_ylabel('Flux', fontsize=22)
        for ax in [ax1b, ax2b]:
            ax.axhline(0, color='y', ls='-', alpha=0.35, lw=1)
            ax.set_ylabel('Residual/Flux', fontsize=18)
            ax.xaxis.set_minor_locator(MultipleLocator(20))
        ax2b.set_xlabel('wavelength', fontsize=20)

        return fig, ax1, ax2, ax1b, ax2b



    def get_corner_mcmcSample(obj, pltlabel = ['logtrans', 'logage', 'zH']):
        for pi, plabel in enumerate(pltlabel):
            if pi ==0:
                mcmcSamples_tem = np.copy(obj.mcmc[plabel])
            else:
                mcmcSamples_tem = np.vstack(( mcmcSamples_tem, obj.mcmc[plabel] ))
        return mcmcSamples_tem.T


    def get_corner_range(obj1, obj2, pltlabel = ['logtrans', 'ML_r', 'MW_r', 'MLMW_r', 'logage', 'zH']):
        outrange = []
        for pi, plabel in enumerate(pltlabel):
            rangetem = (np.min([np.nanmin(obj1.mcmc[plabel]),
                      np.nanmin(obj2.mcmc[plabel])])-0.1,
              np.max([np.nanmax(obj1.mcmc[plabel]),
                      np.nanmax(obj2.mcmc[plabel])])+0.1)
            outrange.append(rangetem)

        return outrange
