# Make a simple spectrum plot

import cafana
import matplotlib.pyplot as plt
import ROOT
import os

# Environment variables and wildcards work. Most commonly you want a SAM
# dataset. Pass -ss --limit 1 on the cafe command line to make this take a
# reasonable amount of time for demo purposes.

for i in range(1,11):
    fname = "/unix/nova/sam_datasets/prod_sumdecaf_R17-11-14-prod4reco.d_nd_genie_nonswap_fhc_nova_v08_full_v1_numu2018/prod_sumdecaf_R17-11-14-prod4reco.d_nd_genie_nonswap_fhc_nova_v08_period1_v1_numu2018_{}_of_10.root".format(i)

    loader = cafana.SpectrumLoader(fname)

    bins = cafana.Binning.Simple(100, 0, 2)

    # Arbitrary code to extract value from StandardRecord. Must be in C++ like this
    kTrackLen = cafana.CVar('''
if(sr.trk.kalman.ntracks == 0) return 0.0f;
return float(sr.trk.kalman.tracks[0].len);
''')

    MC_Truth = cafana.CVar('''
return float(sr.mc.nu[0].x);
''')

    MC_mode = cafana.CVar('''
return float(sr.mc.nu[0].mode);
''')

    kCrudeMuonSel  = cafana.CCut('''
if(sr.trk.kalman.ntracks == 0) return false;
return bool(sr.trk.kalman.tracks[0].len > 200);
''')

    # Spectrum to be filled from the loader
    length = cafana.Spectrum("{}".format(MC_mode), bins, loader, MC_Truth, cafana.kIsNumuCC)

    # Do it!
    loader.Go()
    
    # How to scale histograms
    pot = 18e20

    # We have histograms
    hist_plot = length.ToTH1(pot)    
    h =  hist_plot.Draw()

    os.chdir('/home/ishokar/plots/')

    myfile =ROOT. TFile( "MC_Plot_{}_of_10.pdf".format(i), 'RECREATE' )

    hist_plot.Write()


    # Output mode
    print('Mode value is', MC_mode)

