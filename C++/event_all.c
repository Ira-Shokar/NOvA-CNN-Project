#include "CAFAna/Core/EventList.h"

#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/SpectrumLoader.h"

#include "StandardRecord/Proxy/SRProxy.h"

#include <iostream>

#include "TFile.h"
#include "TTree.h"

namespace ana
{
  /// Helper class for \ref MakeTextListFile
  class ASCIIMaker: public SpectrumLoader
  {
  public:
    ASCIIMaker(const std::string& wildcard,
               const std::vector<Cut>& cut,
               const std::vector<FILE*>& f,
               const std::vector<const Var*>& floatVars,
               const std::vector<const Var*>& intVars)
      : SpectrumLoader(wildcard, kBeam),
        fCut(cut),
        fFile(f),
        fFloatVars(floatVars),
        fIntVars(intVars),
        fNPassed(0)
    {
    }

    ASCIIMaker(const std::string& wildcard,
               const std::vector<Cut>& cut,
               const std::vector<FILE*>& f,
               const std::vector<const MultiVar*>& multivars,
               const std::vector<const MultiVar*>& /*multiIntVars*/)
      : SpectrumLoader(wildcard, kBeam),
        fCut(cut),
        fFile(f),
        fMultiVars(multivars),
        fNPassed(0)
    {
    }

    ASCIIMaker(const std::vector<std::string>& fnames,
               const std::vector<Cut>& cut,
               const std::vector<FILE*>& f,
               const std::vector<const Var*>& floatVars,
               const std::vector<const Var*>& intVars)
      : SpectrumLoader(fnames, kBeam),
        fCut(cut),
        fFile(f),
        fFloatVars(floatVars),
        fIntVars(intVars),
        fNPassed(0)
    {
    }

    ~ASCIIMaker()
    {
      std::cout << "Selected " << fNPassed << " slices." << std::endl;
    }

    void HandleRecord(caf::SRProxy* sr) override
    {
      if(fSpillCut && !(*fSpillCut)(&sr->spill)) return;

      for(unsigned int ic = 0; ic < fCut.size(); ic++){
        if(!fCut[ic](sr))
          continue;

        ++fNPassed;

        for(const Var* v: fIntVars){
          fprintf(fFile[ic], "%d ", int(std::round((*v)(sr))));
        }

        for(const Var* v: fFloatVars){
          fprintf(fFile[ic], "%g ", (*v)(sr));
        }

	for(const MultiVar* mv: fMultiVars){
          for(const double md: (*mv)(sr)){
	    fprintf(fFile[ic], "%g ", md);
          }
	}

        fprintf(fFile[ic], "\n");
      }// end loop over cuts
    }
  protected:
    std::vector<Cut> fCut;
    std::vector<FILE*> fFile;
    std::vector<const Var*> fFloatVars;
    std::vector<const Var*> fIntVars;
    std::vector<const MultiVar*> fMultiVars;
    int fNPassed;
  };

  /// Helper class for \ref MakeEventTTreeFile
  class TreeMaker: public SpectrumLoader
  {
  public:
    TreeMaker(const std::string& wildcard,
              const std::vector<std::pair<std::string, Cut>>& cuts,
              const std::vector<std::pair<std::string, Var>>& floatVars,
              const std::vector<std::pair<std::string, Var>>& intVars)
      : SpectrumLoader(wildcard, kBeam),
        fFloats(floatVars.size()), fInts(intVars.size())
    {
      for(unsigned int ic = 0; ic < cuts.size(); ++ic){
        fCuts.push_back(cuts[ic].second);
        fTrees.push_back(new TTree(cuts[ic].first.c_str(),
                                   cuts[ic].first.c_str()));
        for(unsigned int iv = 0; iv < floatVars.size(); ++iv){
          fFloatVars.push_back(floatVars[iv].second);
          fTrees.back()->Branch(floatVars[iv].first.c_str(), &fFloats[iv]);
        }
        for(unsigned int iv = 0; iv < intVars.size(); ++iv){
          fIntVars.push_back(intVars[iv].second);
          fTrees.back()->Branch(intVars[iv].first.c_str(), &fInts[iv]);
        }
      }
    }

    void HandleRecord(caf::SRProxy* sr) override
    {
      if(fSpillCut && !(*fSpillCut)(&sr->spill)) return;

      bool any = false;

      for(unsigned int ic = 0; ic < fCuts.size(); ++ic){
        if(!fCuts[ic](sr)) continue;

        if(!any){
          any = true;
          for(unsigned int iv = 0; iv < fFloatVars.size(); ++iv){
            fFloats[iv] = fFloatVars[iv](sr);
          }
          for(unsigned int iv = 0; iv < fIntVars.size(); ++iv){
            fInts[iv] = fIntVars[iv](sr);
          }
        }

        fTrees[ic]->Fill();
      }
    }

    void Write()
    {
      for(TTree* tr: fTrees) tr->Write();
    }

  protected:
    std::vector<Cut> fCuts;
    std::vector<Var> fFloatVars;
    std::vector<Var> fIntVars;
    std::vector<TTree*> fTrees;
    std::vector<float> fFloats;
    std::vector<int> fInts;
  };

  //----------------------------------------------------------------------
  //
  // All the variants below have the same bodies (here), but slightly different
  // arguments. We have T=wildcard/file list and U=Var/MultiVar
  template<class T, class U>
  void MakeTextListFileHelper(const T& source,
                              const std::vector<Cut>& cut,
                              const std::vector<std::string>& output,
                              const std::vector<U>& floatVars,
                              const std::vector<U>& intVars,
                              const SpillCut* spillCut)
  {
    assert(output.size() == cut.size());

    std::vector<FILE*> files;
    for(const std::string& out: output)
      files.push_back(fopen(out.c_str(), "w"));

    ASCIIMaker maker(source, cut, files, floatVars, intVars);
    if(spillCut) maker.SetSpillCut(*spillCut);
    maker.Go();

    for(const std::string& out: output)
      std::cout << "Wrote text list file " << out << std::endl;

    for(FILE* f: files) fclose(f);
  }

  //----------------------------------------------------------------------
  void MakeTextListFile(const std::string& wildcard,
			const std::vector<Cut>& cut,
			const std::vector<std::string>& output,
			const std::vector<const Var*>& floatVars,
			const std::vector<const Var*>& intVars,
			const SpillCut* spillCut)
  {
    MakeTextListFileHelper(wildcard, cut, output, floatVars, intVars, spillCut);
  }

  //----------------------------------------------------------------------

  void MakeTextListFile(const std::vector<std::string>& fnames,
			const std::vector<Cut>& cut,
			const std::vector<std::string>& output,
			const std::vector<const Var*>& floatVars,
			const std::vector<const Var*>& intVars,
			const SpillCut* spillCut)
  {
    MakeTextListFileHelper(fnames, cut, output, floatVars, intVars, spillCut);
  }

  //----------------------------------------------------------------------
  void MakeTextListFile(const std::string& wildcard,
                        const std::vector<Cut>& cut,
                        const std::vector<std::string>& output,
                        const std::vector<const MultiVar*>& multivars,
                        const SpillCut* spillCut)
  {
    MakeTextListFileHelper(wildcard, cut, output, multivars, {}, spillCut);
  }

  //----------------------------------------------------------------------
  //
  // T can be a wildcard or list of file names
  template<class T>
  void MakeEventListFileHelper(const T& source,
                               const std::vector<Cut>& cuts,
                               const std::vector<std::string>& outputs,
                               bool includeSliceIndex,
                               bool includeSliceTime,
                               bool includeCycleNumber,
                               const SpillCut* spillCut,
                               bool includeBatchNumber)
  {
    assert(cuts.size() == outputs.size());

    std::vector<const Var*> intVars = {new SIMPLEVAR(hdr.run),
                                       new SIMPLEVAR(hdr.subrun)};

    if(includeCycleNumber) intVars.push_back(new SIMPLEVAR(hdr.cycle));
    if(includeBatchNumber) intVars.push_back(new SIMPLEVAR(hdr.batch));
    intVars.push_back(new SIMPLEVAR(hdr.evt));
    if(includeSliceIndex) intVars.push_back(new SIMPLEVAR(hdr.subevt));

    std::vector<const Var*> floatVars;
    if(includeSliceTime) floatVars.push_back(new SIMPLEVAR(hdr.subevtmeantime));

    MakeTextListFile(source, cuts, outputs, floatVars, intVars, spillCut);
  }

  //----------------------------------------------------------------------
  void MakeEventListFile(const std::string& wildcard,
                         const std::vector<Cut>& cuts,
                         const std::vector<std::string>& outputs,
                         bool includeSliceIndex,
                         bool includeSliceTime,
                         bool includeCycleNumber,
                         const SpillCut* spillCut,
			 bool includeBatchNumber)
  {
    MakeEventListFileHelper(wildcard, cuts, outputs,
                            includeSliceIndex, includeSliceTime, includeCycleNumber, spillCut, includeBatchNumber);
  }

  //----------------------------------------------------------------------
  void MakeEventListFile(const std::vector<std::string>& fnames,
                         const std::vector<Cut>& cuts,
                         const std::vector<std::string>& outputs,
                         bool includeSliceIndex,
                         bool includeSliceTime,
                         bool includeCycleNumber,
                         const SpillCut* spillCut,
			 bool includeBatchNumber)
  {
    MakeEventListFileHelper(fnames, cuts, outputs,
                            includeSliceIndex,
                            includeSliceTime,
                            includeCycleNumber,
                            spillCut,
                            includeBatchNumber);
  }

  //----------------------------------------------------------------------
  void MakeEventListFile(const std::string& wildcard,
                         const Cut& cut,
                         const std::string& output,
                         bool includeSliceIndex,
                         bool includeSliceTime,
                         bool includeCycleNumber,
                         const SpillCut* spillCut,
			 bool includeBatchNumber)
  {
    MakeEventListFileHelper(wildcard, {cut}, {output},
                            includeSliceIndex,
                            includeSliceTime,
                            includeCycleNumber,
                            spillCut,
                            includeBatchNumber);
  }

  //----------------------------------------------------------------------
  void MakeEventListFile(const std::vector<std::string>& fnames,
                         const Cut& cut,
                         const std::string& output,
                         bool includeSliceIndex,
                         bool includeSliceTime,
                         bool includeCycleNumber,
                         const SpillCut* spillCut,
			 bool includeBatchNumber)
  {
    MakeEventListFileHelper(fnames, {cut}, {output},
                            includeSliceIndex,
                            includeSliceTime,
                            includeCycleNumber,
                            spillCut,
                            includeBatchNumber);

  }

  //----------------------------------------------------------------------
  void MakeEventTTreeFile(const std::string& wildcard,
                          const std::string& output,
                          const std::vector<std::pair<std::string, Cut>>& cuts,
                          const std::vector<std::pair<std::string, Var>>& floatVars,
                          const std::vector<std::pair<std::string, Var>>& intVars,
                          const SpillCut* spillCut)
  {
    // Scope all the TTrees within this
    TFile fout(output.c_str(), "RECREATE");

    TreeMaker maker(wildcard, cuts, floatVars, intVars);

    if(spillCut) maker.SetSpillCut(*spillCut);
    maker.Go();

    fout.cd();
    maker.Write();
    fout.Close();
    std::cout << "Wrote TTree file " << output << std::endl;
  }

}

void demo0()
{
    const std::string fname = "neardet_genie_nonswap_genierw_fhc_v08_2625_r00011289_s10_c000_N19-N19-02-05_v1_20170322_204739_sim.h5caf.h";
    
    const std::vector<Cut> cut = {"kNumuBasicQuality", "kNumuContainND2017"}
    
    const std::vector<std::string>  output = "event_cuts_mu"
    
    MakeEventListFile(fname, cut, output);
    
}

