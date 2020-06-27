#include "event_header.h"

using namespace ana;

void demo1()
{	
  const std::string fname = "neardet_genie_nonswap_genierw_fhc_v08_2625_r00011289_s10_c000_N19-N19-02-05_v1_20170322_204739_sim.h5caf.h";
  
  const std::vector<Cut> cut = {"kNumuBasicQuality", "kNumuContainND2017"};
    
  const std::string output ="event_cuts_mu";
	
  MakeTextListFile(fname, cut, output);

}

