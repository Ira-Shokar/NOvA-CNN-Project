#include "CAFAna/Core/EventList.h"
    
#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/SpectrumLoader.h"
     
#include "StandardRecord/Proxy/SRProxy.h"
#include <iostream>
    
#include "TFile.h"
#include "TTree.h"
 
namespace ana
{
   void MakeTextListFile(const std::string& wildcard,
                         const std::vector<Cut>& cut,
                         const std::vector<std::string>& output,
                         const std::vector<const Var*>& floatVars,
                         const std::vector<const Var*>& intVars,
                         const SpillCut* spillCut)
   {
     MakeTextListFileHelper("neardet_genie_nonswap_genierw_fhc_v08_2625_r00011289_s10_c000_N19-02-05_v1_20170322_204739_sim.h5caf.h5", {"kNumuBasicQuality", "kNumuContainND2017"}, "event_cuts_mu.rft", floatVars, intVars, spillCut);
   }
}
