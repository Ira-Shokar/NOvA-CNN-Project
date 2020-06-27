#pragma once
     
#include "Rtypes.h"
#include "TAttMarker.h"
    
#include "CAFAna/Core/Cut.h"
#include "CAFAna/Core/Var.h"
#include "CAFAna/Core/MultiVar.h"
#include "CAFAna/Core/EventList.h"
    
#include <string>
    
namespace ana;
int main()

{
    MakeTextListFile("neardet_genie_nonswap_genierw_fhc_v08_2625_r00011289_s10_c000_N19-N19-02-05_v1_20170322_204739_sim.h5caf.h", {"kNumuBasicQuality", "kNumuContainND2017"}, "event_cuts_mu", {}, 0);

}

