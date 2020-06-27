// Make a simple spectrum plot

#include "CAFAna/Core/Binning.h"
#include "CAFAna/Cuts/Cuts.h"
#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/SpectrumLoader.h"
#include "CAFAna/Vars/Vars.h"

#include "StandardRecord/Proxy/SRProxy.h"

#include "TCanvas.h"
#include "TH2.h"

#include <sstream>
#include <string>

using namespace ana;


void demo2()
{
  // Environment variables and wildcards work. Most commonly you want a SAM
  // dataset. Pass -ss --limit 1 on the cafe command line to make this take a
  // reasonable amount of time for demo purposes.
  const std::string fname = "/unix/nova/sam_datasets/prod_sumdecaf_R17-11-14-prod4reco.d_nd_genie_nonswap_fhc_nova_v08_full_v1_numu2018/prod_sumdecaf_R17-11-14-prod4reco.d_nd_genie_nonswap_fhc_nova_v08_period1_v1_numu2018_1_of_10.root";

  SpectrumLoader loader(fname);

  const Binning bins = Binning::Simple(100, 0, 2);

  // Specify variables needed and arbitrary code to extract value from
  // SRProxy
  const Var mcTruth ([](const caf::SRProxy* sr)
                      {
                        return float(sr-> mc.nu[0].x);
			cout<<"Mode Value"<< float(sr-> mc.nu[0].mode);
                      });
  
  const Cut kCrudeMuonSel([](const caf::SRProxy* sr)
                       {       
                        if(sr->trk.kalman.ntracks == 0) return false;
                        return sr->trk.kalman.tracks[0].len > 200;
                       });


  // Spectrum to be filled from the loader
  Spectrum len("MC Truth", bins, loader, mcTruth, kCrudeMuonSel);

  // Do it!
  loader.Go();
  // How to scale histograms
  const double pot = 18e20;

  // We have histograms
  len.ToTH1(pot)->Draw("hist");
 
}


