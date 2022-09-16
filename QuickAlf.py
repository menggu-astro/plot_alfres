## plot any alf result using full mode,
# including a spectrum and a corner plot

import numpy as np
import glob
import astropy, pylab, scipy, os, sys
from astropy.table import Table
import astropy.io.ascii as astro_ascii
from astropy.io import ascii, fits

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from get_alfres import ALFbest
import corner

# ---------------------------------------------------------------- #
def get_corner_mcmcSample(obj,
                          pltlabel = ['logtrans', 'ML_r', 'MW_r', 'MLMW_r', 'logage', 'zH']):
    for pi, plabel in enumerate(pltlabel):
        if pi ==0:
            mcmcSamples_tem = np.copy(obj.mcmc[plabel])
        else:
            mcmcSamples_tem = np.vstack(( mcmcSamples_tem, obj.mcmc[plabel] ))
    return mcmcSamples_tem.T

# ---------------------------------------------------------------- #
def get_corner_range(obj1, obj2, pltlabel = ['logtrans', 'ML_r', 'MW_r', 'MLMW_r', 'logage', 'zH'],
                    ):
    outrange = []
    for pi, plabel in enumerate(pltlabel):
        rangetem = (np.min([np.nanmin(obj1.mcmc[plabel]),
                  np.nanmin(obj2.mcmc[plabel])])-0.1,
          np.max([np.nanmax(obj1.mcmc[plabel]),
                  np.nanmax(obj2.mcmc[plabel])])+0.1)
        outrange.append(rangetem)

    return outrange


# ---------------------------------------------------------------- #
def get_alf_header(infile):
    char='#'
    with open (infile, "r") as myfile:
        temdata=myfile.readlines()

    header = []
    for iline in temdata:
        if iline.split(' ')[0] == char:
            temline = np.array(iline.split(' ')[1:])
            header_item = []
            for iitem in temline:
                if len(iitem.split('\n'))>1:
                    header_item.append(float(iitem.split('\n')[0]))
                else:
                    header_item.append(float(iitem))
            header.append(np.array(header_item))
    return header



# ---------------------------------------------------------------- #
def plotalfspec(obj, cont_norm = False, alf_input_header=None,):

    fig = plt.figure(figsize=(16, 10),facecolor='white')
    ax1 =  fig.add_axes([0.10, 0.68, 0.25, 0.30])
    ax1b = fig.add_axes([0.10, 0.55, 0.25, 0.12])
    ax2 =  fig.add_axes([0.40, 0.68, 0.25, 0.30])
    ax2b = fig.add_axes([0.40, 0.55, 0.25, 0.12])
    ax3 =  fig.add_axes([0.70, 0.68, 0.25, 0.30])
    ax3b = fig.add_axes([0.70, 0.55, 0.25, 0.12])
    ax4 =  fig.add_axes([0.10, 0.18, 0.25, 0.30])
    ax4b = fig.add_axes([0.10, 0.05, 0.25, 0.12])
    ax5 =  fig.add_axes([0.40, 0.18, 0.25, 0.30])
    ax5b = fig.add_axes([0.40, 0.05, 0.25, 0.12])
    ax6 =  fig.add_axes([0.70, 0.18, 0.25, 0.30])
    ax6b = fig.add_axes([0.70, 0.05, 0.25, 0.12])

    ax1.set_xlim(4000, 5000); ax1b.set_xlim(4000, 5000)
    ax2.set_xlim(5000, 6000); ax2b.set_xlim(5000, 6000)
    ax3.set_xlim(6000, 7000); ax3b.set_xlim(6000, 7000)
    ax4.set_xlim(7000, 8000); ax4b.set_xlim(7000, 8000)
    ax5.set_xlim(8000, 8920); ax5b.set_xlim(8000, 8920)
    ax6.set_xlim(9620, 10100);ax6b.set_xlim(9620, 10100)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.xaxis.set_ticks([])
    for ax in [ax1, ax4]: ax.set_ylabel('Normalized Flux', fontsize=20, labelpad=22)
    for ax in [ax1b, ax4b]:
        ax.set_ylabel('Fractional\nResidual', fontsize=20, multialignment='center')
    for ax in [ax4b, ax5b, ax6b]: ax.set_xlabel(r'Rest-frame Wavelength [$\mathrm{\AA}$]', fontsize=22)

    velz = float(obj.minchi2['velz'])
    vc_1 = 1.+velz/scipy.constants.c*1e3
    bestwave = obj.bestspec['col1']/vc_1
    bestspec = obj.bestspec['col2']
    bestflux = obj.bestspec['col3']
    bestcont = obj.bestspec['col5']
    besterr = obj.bestspec['col6']
    bestflux[besterr>1e30] = np.nan

    for iax, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
        if cont_norm == False:
            ax.plot(bestwave, bestflux, color='k', alpha=0.3, lw=0.75, label = 'Data')
            ax.fill_between(bestwave, bestflux - besterr, bestflux + besterr, color='gray', alpha=0.6)
            ax.plot(bestwave, bestspec, color='r', alpha=0.50, ms=0, lw=3.5, label = 'Best-fit Model')

            xmin, xmax = ax.get_xlim()[0], ax.get_xlim()[1]
            temidx = (bestwave <= xmax)&(bestwave>= xmin)
            try:
                ax.set_ylim(np.nanmin(bestspec[temidx])*0.95, np.nanmax(bestspec[temidx])*1.05)
            except:
                pass


        elif cont_norm == True:
            ax.plot(bestwave, bestflux/bestcont, color='k', alpha=1, lw=1.0, label = 'Data')
            ax.fill_between(bestwave,
                            (bestflux - besterr)/bestcont,
                            (bestflux + besterr)/bestcont, color='gray', alpha=0.6)
            ax.plot(bestwave, bestspec/bestcont, color='r', alpha=0.50, ms=0, lw=3.0, label = 'Best Model')

            xmin, xmax = ax.get_xlim()[0], ax.get_xlim()[1]
            temy = bestspec/bestcont
            ax.set_ylim(np.nanmin(temy[(bestwave<xmax)&(bestwave>xmin)])*0.95,
                        np.nanmax(temy[(bestwave<xmax)&(bestwave>xmin)])*1.05)
            ax.yaxis.set_minor_locator(MultipleLocator(0.005))

    ax4.legend(loc=2, fontsize=15)
    for ax in [ax2, ax3, ax4, ax5, ax6]:
        minx, maxx = ax.get_xlim()
        temidx = (bestwave <= maxx)&(bestwave>= minx)
        temwave = bestwave[temidx]
        wavewidth = np.nanmean(temwave[1:]-temwave[:-1])
        print('%.2f' %wavewidth, end=',')
        sn_ = np.nanmedian(bestflux[temidx]/besterr[temidx])/wavewidth
        ax.annotate(r'S/N$_{%.0f-%.0f\AA}=%.0f{\AA^{-1}}$'%(minx, maxx, sn_), fontsize=15, xy=(0.4, 0.05), xycoords ='axes fraction')

    label_ = 'MLMW_r'
    ax1.annotate(r'$\alpha=%.2f^{+%.2f}_{-%.2f}$'%(obj.cl50[label_],obj.cl84[label_]-obj.cl50[label_],
                                                   obj.cl50[label_]-obj.cl16[label_]),
                 fontsize=17, xy=(0.65, 0.05), xycoords ='axes fraction' )

    for i, ax in enumerate([ax1b, ax2b, ax3b, ax4b, ax5b, ax6b]):
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.fill_between(bestwave, y1 = -abs(besterr/bestflux), y2 = abs(besterr/bestflux),
                        color='#fec44f', alpha=0.35)
        ax.plot(bestwave, (bestspec-bestflux)/bestflux , color='k', lw=1.2)
        ax.set_ylim(-0.03, 0.03)
        ax.yaxis.set_minor_locator(MultipleLocator(0.005))
        #ax.autoscale(axis='y', tight=True)
        ax.axhline(0, color='y', ls='-', alpha=0.45, lw=1.5)

    if alf_input_header is not None:
        unique_header = np.unique(alf_input_header) * 1e4
        for ax in [ax1b, ax2b, ax3b, ax4b, ax5b, ax6b]:
            for iheader in unique_header:
                if iheader >= ax.get_xlim()[0] and iheader <= ax.get_xlim()[1]:
                    ax.axvline(iheader, color='orange', ls='--')

    fig.tight_layout()
    fig.savefig('{0}/Spec2_{1}.png'.format(obj.outdir, obj.fullname), bbox_inches='tight',dpi=80 )
    fig.clf()
    del fig



# ---------------------------------------------------------------- #
def plotalfcorner(obj, pltlabel=['chi2', 'velz', 'sigma', 'zH', 'FeH', 'logage',
                          'MgFe','IMF1', 'IMF2', 'IMF3', 'ML_r', 'MLMW_r'],
                mcmclabels=[r'$\chi^2$', 'velz', r'$\sigma$', 'zH', 'FeH', 'logage',
                            'MgFe', 'IMF1', 'IMF2', 'IMF3', r'ML$_r$', r'MLMW$_r$']):

    pltlabel = pltlabel #,
    mcmcSamples1 = get_corner_mcmcSample(pltlabel=pltlabel, obj = obj)
    mcmcUse1 = mcmcSamples1[:,:]
    truvl = []; newrange = []
    for ilabel in pltlabel:
        if ilabel == 'chi2':truvl.append( np.nan)
        else:truvl.append( float(obj.minchi2[ilabel]) )


    for il, ilabel in enumerate(pltlabel):
        newrange.append([np.nanpercentile(obj.mcmc[ilabel], 2.5), np.nanpercentile(obj.mcmc[ilabel], 97.5)])
    fig1 = corner.corner(mcmcUse1, #[mcmcUse1[:,1]>=200]
                         bins=25,color='#e32636',alpha=0.5,
                         smooth=1,truths=truvl,truth_color='#0047ab',
                         range=newrange,labels = mcmclabels,
                         label_kwargs={'fontsize':20},quantiles=[0.16, 0.5, 0.84],
                         plot_contours=True,fill_contours=True,
                         show_titles=True,title_kwargs={"fontsize": 17},
                         hist_kwargs={"histtype": 'stepfilled',"alpha": 0.4,"edgecolor": "none"},use_math_text=True)

    fig1.savefig('{0}/Alfcorner1a_{1}.png'.format(obj.outdir, obj.fullname),
                 bbox_inches='tight',dpi=60 )
    fig1.clf()
    del fig1

    
# ---------------------------------------------------------------- #
def plotalfcorner_imf(obj, pltlabel=['IMF1', 'IMF2', 'IMF3', 'MLMW_r'],
                      mcmclabels=['IMF1', 'IMF2', 'IMF3',r'$\alpha$']):

    pltlabel = pltlabel #,
    mcmcSamples1 = get_corner_mcmcSample(pltlabel=pltlabel, obj = obj)
    mcmcUse1 = mcmcSamples1[:,:]
    truvl = []; newrange = []
    for ilabel in pltlabel:
        if ilabel == 'chi2':truvl.append( np.nan)
        else:truvl.append( float(obj.minchi2[ilabel]) )

    for il, ilabel in enumerate(pltlabel):
        newrange.append([np.nanpercentile(obj.mcmc[ilabel], 2.5), np.nanpercentile(obj.mcmc[ilabel], 97.5)])

    fig1 = corner.corner(mcmcUse1, #[mcmcUse1[:,1]>=200]
                         bins=25,color='#e32636',alpha=0.5,
                         smooth=1,truths=truvl,truth_color='#0047ab',
                         range=newrange,labels = mcmclabels,
                         label_kwargs={'fontsize':20},quantiles=[0.16, 0.5, 0.84],
                         plot_contours=True,fill_contours=True,
                         show_titles=True,title_kwargs={"fontsize": 17},
                         hist_kwargs={"histtype": 'stepfilled',"alpha": 0.4,"edgecolor": "none"},use_math_text=True)

    fig1.savefig('{0}/Alfcorner2_{1}.png'.format(obj.outdir, obj.fullname),
                 bbox_inches='tight',dpi=60 )
    fig1.clf()
    del fig1


# ---------------------------------------------------------------- #
def plothist(in_obj):
    fig, axl = plt.subplots(8, 7, figsize=(30, 24))
    axl = axl.flatten()
    j = 0
    keylist = ['velz', 'sigma', 'logage', 'zH', 'FeH', 'aH', 'CH', 'NH', 'NaH', 'MgH',
               'SiH', 'KH', 'CaH', 'TiH', 'VH', 'CrH', 'MnH', 'CoH', 'NiH', 'CuH', 'SrH',
               'BaH', 'EuH', 'Teff', 'IMF1', 'IMF2', 'logfy', 'sigma2', 'velz2', 'logm7g',
               'hotteff', 'loghot', 'fy_logage', 'logemline_H', 'logemline_Oii', 'logemline_Oiii',
               'logemline_Sii', 'logemline_Ni', 'logemline_Nii', 'logtrans', 'jitter',
               'logsky', 'IMF3', 'IMF4', 'h3', 'h4', 'ML_r', 'ML_i', 'ML_k', 'MW_r', 'MW_i',
               'MW_k', 'MLMW_r', 'MLMW_i']
    for i, ikey in enumerate(keylist):
        min_, max_ = in_obj.mcmc[ikey].min(), in_obj.mcmc[ikey].max()
        minchi2_val = in_obj.minchi2[ikey]
        min_ = min(minchi2_val, min_)
        max_ = max(minchi2_val, max_)
        binsize = abs(max_-min_)/50

        if binsize==0:
            continue
        n, b = np.histogram(in_obj.mcmc[ikey], bins=np.arange(min_, max_, binsize))
        #n = norm_kde(n, 10.)
        x0 = 0.5 * (b[1:] + b[:-1])
        y0 = n
        axl[j].fill_between(x0, y0, color='#6baed6')
        axl[j].axvline(minchi2_val, color='orange', ls='--')
        axl[j].annotate(text=('-').join(ikey.split('_')), xy=(0.1, 0.9), xytext=(0.1, 0.9),
                        xycoords='axes fraction')
        j+=1
    fig.savefig('{0}/Hist1D_{1}.png'.format(in_obj.outdir, in_obj.fullname),
                 bbox_inches='tight',dpi=70 )
    fig.clf()
    del fig

# ---------------------------------------------------------------- #
if __name__ == "__main__":
    
    # ============== lines to update =============== #
    dir_input = '/Users/menggu/alfinput'
    dir_output = '/Users/menggu/alfresults'
    dir_figure = '/Users/menggu/figures/'
    all_sumfile = glob.glob(dir_output + "/ldss3_dr293*n4839*wave6e*_imf1.sum")
    # ================================================ #
    
    all_sumfile = np.unique(all_sumfile)
    print(all_sumfile)
    for i, ifile0 in enumerate(all_sumfile):
        ifile = ifile0.split('/')[-1].split('.sum')[0]
        fname = ('_').join(ifile.split('_')[:-1])
        appdx = ifile.split('_')[-1]
        print(fname, appdx)
        galname = ('').join(fname.split('_')[2:-2])

        tem = ALFbest(respath = dir_output, inpath = dir_input,
                      fname = fname, appdx=appdx, alfmode = 1, outdir = dir_figure)
        tem.libcorrect()
        tem.imfdist()
        image_name = '{0}/'+galname+'/Spec2_{1}.png'.format(tem.outdir, tem.fullname)
        image_name = '{0}/Spec2_{1}.png'.format(tem.outdir, tem.fullname)

        alf_input_header = get_alf_header(dir_input + '/' + fname + '.dat')

        plotalfcorner(tem, 
                      pltlabel=['velz','sigma','FeH','logage','MgFe','NaFe','aFe','CFe','CaFe', 'MLMW_r','ML_r'],
                      mcmclabels=['velz',r'$\sigma$','FeH','logage','MgFe','NaFe','OFe','CFe','CaFe', r'$\alpha$','M/L'],)
        plotalfspec(tem, cont_norm = False, alf_input_header = None)
