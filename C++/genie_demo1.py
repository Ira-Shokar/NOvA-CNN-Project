# Make a simple spectrum plot

import cafana

import matplotlib.pyplot as plt

from ROOT import TCanvas 

# Environment variables and wildcards work. Most commonly you want a SAM
# dataset. Pass -ss --limit 1 on the cafe command line to make this take a
# reasonable amount of time for demo purposes.

for i in range(1,11):
    fname = "/unix/nova/sam_datasets/prod_sumdecaf_R17-11-14-prod4reco.d_nd_genie_nonswap_fhc_nova_v08_full_v1_numu2018/prod_sumdecaf_R17-11-14-prod4reco.d_nd_genie_nonswap_fhc_nova_v08_period1_v1_numu2018_{}_of_10.root".format(i)

    loader = cafana.SpectrumLoader(fname)

    bins = cafana.Binning.Simple(100, 0, 2)


    MC_Truth = cafana.CVar('''
return float(sr.mc.nu[0].x);
''')

    Cut = cafana.CCut( '''
kNumuContainND2017(
     [](const caf::SRProxy* sr)
    { if( !sr->vtx.elastic.IsValid ) return false;
      // reconstructed showers all contained
      for( unsigned int i = 0; i < sr->vtx.elastic.fuzzyk.nshwlid; ++i ) {
        TVector3 start = sr->vtx.elastic.fuzzyk.png[i].shwlid.start;
        TVector3 stop  = sr->vtx.elastic.fuzzyk.png[i].shwlid.stop;
          if( std::min( start.X(), stop.X() ) < -180.0 ) return false;
       if( std::max( start.X(), stop.X() ) >  180.0 ) return false;
       if( std::min( start.Y(), stop.Y() ) < -180.0 ) return false;
        if( std::max( start.Y(), stop.Y() ) >  180.0 ) return false;
       if( std::min( start.Z(), stop.Z() ) <   20.0 ) return false;
        if( std::max( start.Z(), stop.Z() ) > 1525.0 ) return false;
       }

      // only primary muon track present in muon catcher
      if( sr->trk.kalman.ntracks < 1 ) return false;
      for( unsigned int i = 0; i < sr->trk.kalman.ntracks; ++i ) {
        if( i == sr->trk.kalman.idxremid ) continue;
        else if( sr->trk.kalman.tracks[i].start.Z() > 1275 ||
                 sr->trk.kalman.tracks[i].stop.Z()  > 1275 )
          return false;
      }

      return ( sr->trk.kalman.ntracks > sr->trk.kalman.idxremid
               && sr->slc.firstplane > 1   // skip 0 and 1
               && sr->slc.lastplane  < 212 // skip 212 and 213
              && sr->trk.kalman.tracks[0].start.Z() < 1100
                 // vertex definitely outside mC
            && ( sr->trk.kalman.tracks[0].stop.Z() < 1275
                    || sr->sel.contain.kalyposattrans < 55 ) // air gap
               && sr->sel.contain.kalfwdcellnd > 5
               && sr->sel.contain.kalbakcellnd > 10 );
       }
   );''')

    # Spectrum to be filled from the loader
    length = cafana.Spectrum(bins, loader, MC_Truth, cafana.kIsNumuCC);

    # Do it!
    loader.Go()
    
    # How to scale histograms
    pot = 18e20

    # We have histograms
    hist_plot = length.ToTH1(pot).Draw('hist')
    
    def histogram_plot(hist, outFile):

        # initialisation 
        # gROOT.SetStyle("Plain") ;
        # gStyle.SetOptStat(0)
        # gStyle.SetTitleXOffset (1.25);
        # gStyle.SetTitleYOffset (1.5);

        c1 = TCanvas('title', 'name', 600, 500)
        c1.SetTicks(1,1);
        #c1.SetBottomMargin(0.3);
        c1.SetLeftMargin(0.2);   
        c1.SetLogy(1); 
   
        key_list = hist.keys()
        nhist = len(key_list)

        for i in range(nhist):
            key	 = key_list[i]
            fName = outFile
            if i == 0: fName = outFile + '('
            if i == nhist - 1: fName = outFile + ')'
            hist[key].Draw()
            c1.Print(fName)

        return 	
    
    histogram_plot(hist_plot, "3_10_10_plot_{}_of_10.pdf".format(i))
    
    # Output mode
    print('Mode value is', MC_mode)
	
pp.close()
