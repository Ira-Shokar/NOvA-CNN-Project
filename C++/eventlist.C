#pragma once

#include "CAFAna/Core/Var.h"
#include "CAFAna/Core/Cut.h"
#include "CAFAna/Core/EventList.h"
#include "CAFAna/Vars/GiBUUWeights.h"

using namespace ana;

void eventlist()
{      
    MakeEventListFile("/unix/nova/sam_datasets/prod_flatsumdecaf_R17-11-14-prod4reco.i_nd_gibuu_nonswap_fhc_nova_v08_full_v1_numu2018/*",
                     {kGibuuWeight},
                     {"gibuu_weights_list"}, "proxy", 0);
}


