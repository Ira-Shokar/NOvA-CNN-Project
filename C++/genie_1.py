# Make a simple spectrum plot

import cafana

import matplotlib.pyplot as plt

import ROOT

# Environment variables and wildcards work. Most commonly you want a SAM
# dataset. Pass -ss --limit 1 on the cafe command line to make this take a
# reasonable amount of time for demo purposes.

for i in range(1,1):
    fname = '/unix/nova/sam_datasets/prod_flatsumdecaf_R17-11-14-prod4reco.d_nd_genie_nonswap_fhc_nova_v08_full_v1_numu2018/prod_flatsumdecaf_R17-11-14-prod4reco.d_nd_genie_nonswap_fhc_nova_v08_period1_v1_numu2018_1_of_10.root'

    loader = cafana.SpectrumLoader(fname)

    bins = cafana.Binning.Simple(100, 0, 2)

    # Arbitrary code to extract value from StandardRecord. Must be in C++ like this

    MC_Truth = cafana.CVar('''
return float(sr.mc.nu[0].x);
''')

    kNumuCutND2018  = cafana.CCut('''
return (sr->energy.numu.trkccE > 0 && // nothing is terribly wrong
sr->sel.remid.pid > 0   &&    // ensures at least 1 3D Kalman track with a remid value
sr->slc.nhit > 20       &&    // low hits stuff is junk
sr->slc.ncontplanes > 4 &&    // remove really vertical stuff
sr->trk.cosmic.ntracks > 0 ); // need a cosmic track
''')

    # Spectrum to be filled from the loader
    length = cafana.Spectrum("{}".format(MC_mode), bins, loader, MC_Truth, kNumuCutND2018)

    # Do it!
    loader.Go()
    
    # How to scale histograms
    pot = 18e20

    # We have histograms
    hist_plot = length.ToTH1(pot)
    h =  hist_plot.Draw('hist')
    h.savefig("MC_Plot_{}_of_10.pdf".format(i), bbox_inches='tight')

    # Output mode
    print('Mode value is', MC_mode)

    break
