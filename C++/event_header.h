#pragma once

#include "Rtypes.h"
#include "TAttMarker.h"

#include "CAFAna/Core/Cut.h"
#include "CAFAna/Core/Var.h"
#include "CAFAna/Core/MultiVar.h"

#include <string>

namespace ana
{
  /// \brief Make a file listing all the events passing the specified cut
  ///
  /// \param wildcard Filename or wildcard to use for input
  /// \param cut      The cut events must pass
  /// \param output   The filename for the output.
  /// \param floatVars Vector with the vars to be included in
  ///                  the output
  /// \param intVars  Vars whose output should be formatted with as integers
  /// \param spillCut Optional requirement on SRSpill
  void MakeTextListFile(const std::string& wildcard,
                        const std::vector<Cut>& cut,
                        const std::vector<std::string>& output,
                        const std::vector<const Var*>& floatVars,
                        const std::vector<const Var*>& intVars,
                        const SpillCut* spillCut = 0);

  void MakeTextListFile(const std::string& wildcard,
                        const std::vector<Cut>& cut,
                        const std::vector<std::string>& output,
                        const std::vector<const Var*>& floatVars,
                        const SpillCut* spillCut = 0)
  {
    MakeTextListFile(wildcard, cut, output, floatVars, {}, spillCut);
  }

  /// \brief Make a file listing all the events passing the specified cut
  ///
  /// \param fnames   Vector of filenames to use for input
  /// \param cut      The cut events must pass
  /// \param output   The filename for the output.
  /// \param floatVars Vector with the vars to be included in
  ///                  the output
  /// \param intVars  Vars whose output should be formatted as integers
  /// \param spillCut Optional requirement on SRSpill
  void MakeTextListFile(const std::vector<std::string>& fnames,
                        const std::vector<Cut>& cut,
                        const std::vector<std::string>& output,
                        const std::vector<const Var*>& floatVars,
                        const std::vector<const Var*>& intVars,
                        const SpillCut* spillCut = 0);

  void MakeTextListFile(const std::vector<std::string>& fnames,
                        const std::vector<Cut>& cut,
                        const std::vector<std::string>& output,
                        const std::vector<const Var*>& floatVars,
                        const SpillCut* spillCut = 0)
  {
    MakeTextListFile(fnames, cut, output, floatVars, {}, spillCut);
  }

  /// \brief Make a file listing multi-var values
  void MakeTextListFile(const std::string& wildcard,
                        const std::vector<Cut>& cut,
                        const std::vector<std::string>& output,
                        const std::vector<const MultiVar*>& multivars = {},
                        const SpillCut* spillCut = 0);

  /// \brief Make a file listing all the events passing the specified cut
  ///
  /// \param wildcard Filename or wildcard to use for input
  /// \param cut      The cut events must pass
  /// \param output   The filename for the output.
  ///                 Format is space-seperated run, subrun, event.
  /// \param includeSliceIndex  Add a column for the slice index
  /// \param includeSliceTime   Add a column for the slice mean time (ns)
  /// \param includeCycleNumber Add a column for the cycle number.
  ///                           Cycle number differentiates multiple
  ///                           instances of a given run/subrun combination
  ///                           during file production.
  /// \param spillCut           Optional requirement on SRSpil
  /// \param includeBatchNumber Add a column for the batch number.
  ///                           Differentiates different batches that end
  ///                           up with the same run/subrun/cycle combo.
  void MakeEventListFile(const std::string& wildcard,
                         const Cut& cut,
                         const std::string& output,
                         bool includeSliceIndex   = false,
                         bool includeSliceTime    = false,
                         bool includeCycleNumber  = false,
                         const SpillCut* spillCut = 0,
			 bool includeBatchNumber  = false);

  /// \brief Make a set of files listing all the events passing cuts
  /// One output file is created for each cut in the cuts vector
  /// \param wildcard Filename or wildcard to use for input
  /// \param cuts     A vector of cuts events must pass
  /// \param outputs  The filenames for the output.
  ///                 Format is space-seperated run, subrun, event.
  /// \param includeSliceIndex Add a column for the slice index
  /// \param includeSliceTime  Add a column for the slice mean time (ns)
  /// \param includeCycleNumber Add a column for the cycle number.
  ///                           Cycle number differentiates multiple
  ///                           instances of a given run/subrun combination
  ///                           during file production.
  /// \param spillCut          Optional requirement on SRSpill
  /// \param includeBatchNumber Add a column for the batch number.
  ///                           Differentiates different batches that end
  ///                           up with the same run/subrun/cycle combo.
  void MakeEventListFile(const std::string& wildcard,
                         const std::vector<Cut>& cuts,
                         const std::vector<std::string>& outputs,
                         bool includeSliceIndex   = false,
                         bool includeSliceTime    = false,
                         bool includeCycleNumber  = false,
                         const SpillCut* spillCut = 0,
			 bool includeBatchNumber  = false);

  /// \brief Make a set of files listing all the events passing cuts
  /// One output file is created for each cut in the cuts vector
  /// \param wildcard Vector of filenames or wildcards to use for input
  /// \param cuts     The cut events must pass
  /// \param outputs  The filename for the output.
  ///                 Format is space-seperated run, subrun, event.
  /// \param includeSliceIndex Add a column for the slice index
  /// \param includeSliceTime  Add a column for the slice mean time (ns)
  /// \param includeCycleNumber Add a column for the cycle number.
  ///                           Cycle number differentiates multiple
  ///                           instances of a given run/subrun combination
  ///                           during file production.
  /// \param spillCut          Optional requirement on SRSpill
  /// \param includeBatchNumber Add a column for the batch number.
  ///                           Differentiates different batches that end
  ///                           up with the same run/subrun/cycle combo.
  void MakeEventListFile(const std::vector<std::string>& fnames,
                         const Cut& cut,
                         const std::string& output,
                         bool includeSliceIndex   = false,
                         bool includeSliceTime    = false,
                         bool includeCycleNumber  = false,
                         const SpillCut* spillCut = 0,
			 bool includeBatchNumber  = false);

  /// \brief Make a set of files listing all the events passing cuts
  /// One output file is created for each cut in the cuts vector
  /// \param fnames   Vector of filenames or wildcards to use for input
  /// \param cuts     A vector of cuts events must pass
  /// \param outputs  The filenames for the output.
  ///                 Format is space-seperated run, subrun, event.
  /// \param includeSliceIndex Add a column for the slice index
  /// \param includeSliceTime  Add a column for the slice mean time (ns)
  /// \param includeCycleNumber Add a column for the cycle number.
  ///                           Cycle number differentiates multiple
  ///                           instances of a given run/subrun combination
  ///                           during file production.
  /// \param spillCut          Optional requirement on SRSpill
  /// \param includeBatchNumber Add a column for the batch number.
  ///                           Differentiates different batches that end
  ///                           up with the same run/subrun/cycle combo.
  void MakeEventListFile(const std::vector<std::string>& fnames,
                         const std::vector<Cut>& cuts,
                         const std::vector<std::string>& outputs,
                         bool includeSliceIndex   = false,
                         bool includeSliceTime    = false,
                         bool includeCycleNumber  = false,
                         const SpillCut* spillCut = 0,
			 bool includeBatchNumber  = false);

  /// \brief Make a ROOT file listing all the events passing the specified cut
  ///
  /// \param wildcard Filename or wildcard to use for input
  /// \param cuts     Cuts events must pass, with tree name
  /// \param vars     Vector with the vars to be included in
  ///                 the output, with branch name
  /// \param output   The filename for the output.
  /// \param spillCut Optional requirement on SRSpill
  void MakeEventTTreeFile(const std::string& wildcard,
                          const std::string& output,
                          const std::vector<std::pair<std::string, Cut>>& cuts,
                          const std::vector<std::pair<std::string, Var>>& floatVars,
                          const std::vector<std::pair<std::string, Var>>& intVars = {},
                          const SpillCut* spillCut = 0);
}

