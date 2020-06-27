// Make a simple spectrum plot

#include "CAFAna/Core/Binning.h"
#include "CAFAna/Cuts/Cuts.h"
#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/SpectrumLoader.h"
#include "CAFAna/Vars/Vars.h"

#include "StandardRecord/Proxy/SRProxy.h"

#include "TCanvas.h"
#include "TH2.h"

#include <iterator>

using namespace ana;


void demo_jan()
{
  // Environment variables and wildcards work. Most commonly you want a SAM
  // dataset. Pass -ss --limit 1 on the cafe command line to make this take a
  // reasonable amount of time for demo purposes.
  const std::string fname = "/unix/nova/sam_datasets/prod_flatsumdecaf_R17-11-14-prod4reco.d_nd_genie_nonswap_fhc_nova_v08_full_v1_numu2018/prod_flatsumdecaf_R17-11-14-prod4reco.d_nd_genie_nonswap_fhc_nova_v08_period5_v1_numu2018_9_of_70.root";

  SpectrumLoader loader(fname);

  const Binning bins = Binning::Simple(100, 0, 1000);

  // Specify variables needed and arbitrary code to extract value from
  // SRProxy
  const Var kTrackLen([](const caf::SRProxy* sr)
                      {
                        if(sr->trk.kalman.ntracks == 0) return 0.0f;
                        return float(sr->trk.kalman.tracks[0].len);
                      });

  const Var RecEnergyNuMu([](const caf::SRProxy* sr)
                      {
                        if(sr->mc.nu[0].nhitslc == 0) return 0.0f;
                        return float(sr->mc.nu[0].E);
                      });


  const Var MC_Truth([](const caf::SRProxy* sr)
                      {
                       return float(sr->mc.nu[0].x);
                      });

  const Cut kNumuCutND2018([](const caf::SRProxy* sr)
                      {
                       return (sr->energy.numu.trkccE > 0 && // nothing is terribly wrong
                       sr->sel.remid.pid > 0   &&    // ensures at least 1 3D Kalman track with a remid value
                       sr->slc.nhit > 20	&&    // low hits stuff is junk
                       sr->slc.ncontplanes > 4 &&    // remove really vertical stuff
                       sr->trk.cosmic.ntracks > 0 ); // need a cosmic track
                       });
 
  const Cut kNumuContainND2017([](const caf::SRProxy* sr)
                      { 
//                       if( !sr->vtx.elastic.IsValid ) return false;
  //                     // reconstructed showers all contained
    //                   for( unsigned int i = 0; i < sr->vtx.elastic.fuzzyk.nshwlid; ++i ) {
      //                 TVector3 start = sr->vtx.elastic.fuzzyk.png[i].shwlid.start;
        //               TVector3 stop  = sr->vtx.elastic.fuzzyk.png[i].shwlid.stop;
          //             if( std::min( start.X(), stop.X() ) < -180.0 ) return false;
            //           if( std::max( start.X(), stop.X() ) >  180.0 ) return false;
              //         if( std::min( start.Y(), stop.Y() ) < -180.0 ) return false;
                //       if( std::max( start.Y(), stop.Y() ) >  180.0 ) return false;
                  //     if( std::min( start.Z(), stop.Z() ) <   20.0 ) return false;
                    //   if( std::max( start.Z(), stop.Z() ) > 1525.0 ) return false;
                      // }

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
                      });

  // Spectrum to be filled from the loader
  Spectrum len("Reconstructed Neutrino Energy (Gev)", bins, loader, kTrackLen, kNumuContainND2017);

  // Do it!
  loader.Go();

  // How to scale histograms
  const double pot = 18e20;

  // We have histograms
  len.ToTH1(pot)->Draw("hist");
}
