// Make a simple spectrum plot

#include "CAFAna/Core/Binning.h"
#include "CAFAna/Cuts/Cuts.h"
#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/SpectrumLoader.h"
#include "CAFAna/Vars/Vars.h"

#include "StandardRecord/Proxy/SRProxy.h"

#include "TCanvas.h"
#include "TH2.h"

using namespace ana;


void demo0()
{
  // Environment variables and wildcards work. Most commonly you want a SAM
  // dataset. Pass -ss --limit 1 on the cafe command line to make this take a
  // reasonable amount of time for demo purposes.
  const std::string fname = "prod_sumdecaf_R17-03-01-prod3reco.k_fd_genie_nonswap_fhc_nova_v08_full_nue_or_numu_or_nus_contain_v1";

  SpectrumLoader loader(fname);

  const Binning bins = Binning::Simple(100, 0, 1000);

  // Specify variables needed and arbitrary code to extract value from
  // SRProxy
  const Var kTrackLen([](const caf::SRProxy* sr)
                      {
                        if(sr->trk.kalman.ntracks == 0) return 0.0f;
                        return float(sr->trk.kalman.tracks[0].len);
                      });

  // Spectrum to be filled from the loader
  Spectrum len("Track length (cm)", bins, loader, kTrackLen, kIsNumuCC);

  // Do it!
  loader.Go();

  // How to scale histograms
  const double pot = 18e20;

  // We have histograms
  len.ToTH1(pot)->Draw("hist");
}
