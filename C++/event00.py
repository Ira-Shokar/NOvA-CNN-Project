import cafana
import ROOT
fname = "neardet_genie_nonswap_genierw_fhc_v08_2625_r00011289_s10_c000_N19-N19-02-05_v1_20170322_204739_sim.h5caf.h"
cut = ("kNumuBasicQuality", "kNumuContainND2017")
out = "event_cuts_mu"
list = cafana.MakeEventListFile(fname, cut, out)
