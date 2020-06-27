#pragma once
     
#include "Rtypes.h"
#include "TAttMarker.h"
    
#include "CAFAna/Core/Cut.h"
#include "CAFAna/Core/Var.h"
#include "CAFAna/Core/MultiVar.h"
    
#include <string>
    
namespace ana
{
   void MakeTextListFile(const std::vector<std::string>& fnames,
                         const std::vector<Cut>& cut,
                         const std::vector<std::string>& output,
                         const std::vector<const Var*>& floatVars,
                         const SpillCut* spillCut = 0)
}

{
  MakeTextListFile("neardet_genie_nonswap_genierw_fhc_v08_2625_r00011289_s10_c000_N19-02-05_v1_20170322_204739_sim.h5caf.h5", {"kNumuBasicQuality", "kNumuContainND2017"}, "event_cuts_mu.rft", floatVars, spillCut);
}


