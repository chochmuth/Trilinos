/*
// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER
*/

#include "Teuchos_UnitTestHarness.hpp"

#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Export.hpp"
#include "Teuchos_CommHelpers.hpp"

#include <iostream>
#include <numeric>
#include <vector>

// Test for Trilinos GitHub MyIssue.

namespace { // (anonymous)

  using GST = Tpetra::global_size_t;
  using LO = Tpetra::Map<>::local_ordinal_type;

  // Try to get a 64-bit GlobalOrdinal type; that should be more
  // likely to manifest atomic update issues.  If we can't, though,
  // then a 32-bit type is fine.  long long is guaranteed to be at
  // least 64 bits, so prefer that over long or unsigned long.
#if defined(HAVE_TPETRA_INST_INT_LONG_LONG)
  using GO = long long; // always at least 64 bits
#elif defined(HAVE_TPETRA_INST_INT_LONG)
  using GO = long; // may be 64 or 32 bits
#elif defined(HAVE_TPETRA_INST_INT_UNSIGNED_LONG)
  using GO = unsigned long; // may be 64 or 32 bits
#else
  using GO = Tpetra::Map<>::global_ordinal_type;
#endif
  using SC = double;
    
  using map_type = Tpetra::Map<LO, GO>;
  using vec_type = Tpetra::Vector<SC, LO, GO>;
  using mat_type = Tpetra::CrsMatrix<SC, LO, GO>;
  using export_type = Tpetra::Export<LO, GO>;

  Teuchos::RCP<const map_type>
  createTargetMap (const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
  {
    const GST gblNumInds = static_cast<GST> (comm->getSize ());
    const GO indexBase = 0;

    return Teuchos::rcp (new map_type (gblNumInds, indexBase, comm));
  }

Teuchos::RCP<const map_type>
createTargetMap2 (const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
{
//    const GST gblNumInds = static_cast<GST> (comm->getSize ());
//    const GO indexBase = 0;
//    const GST usedProcs = static_cast<GST> (comm->getSize() - 2);
//    GO numMyRows = 0;
//    if (comm->getRank()<usedProcs) {
//        numMyRows++;
//    }
//    std::cout << "gblNumInds%usedProcs:"<< gblNumInds%usedProcs << std::endl;
//    if (comm->getRank()< (gblNumInds%usedProcs) ) {
//        numMyRows++;
//    }
//    if ((gblNumInds%usedProcs)==0) {
//        if (comm->getSize()>usedProcs && comm->getRank()< usedProcs) {
//            for (unsigned i=0; i<unsigned(gblNumInds/usedProcs/usedProcs); i++) {
//                numMyRows++;
//            }
//        }
//    }
//    std::cout << "numMyRows:"<< numMyRows << std::endl;

    Teuchos::Array<  GO > indexList;
    LO numMyRows = 0;
    if(comm->getRank()==0){
        indexList.push_back(0);
        indexList.push_back(1);
        numMyRows = 2;
    }
    if(comm->getRank()==1){
        
    }
    if(comm->getRank()==2){
        
    }
    if(comm->getRank()==3){
        indexList.push_back(2);
        numMyRows = 1;
    }
    if(comm->getRank()==4){
        
    }
    if(comm->getRank()==5){
        
    }
    if(comm->getRank()==6){
        indexList.push_back(3);
        numMyRows = 1;
    }
    if(comm->getRank()==7){
        
    }
    if(comm->getRank()==8){
        
    }
    const GO indexBase = 0;
//    return Teuchos::rcp (new map_type (-1, indexList(), indexBase, comm));
    return Teuchos::rcp (new map_type (Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), numMyRows, indexBase, comm));
}

  // As overlapping as possible, but avoid the explicitly "locally
  // replicated" path, just in case Tpetra optimizes for that (it
  // doesn't really, but it could).
  Teuchos::RCP<const map_type>
  createSourceMap (const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
  {
    const GST flag = Teuchos::OrdinalTraits<GST>::invalid ();
    const LO lclNumInds = static_cast<LO> (comm->getSize ());
    std::vector<GO> gblInds (lclNumInds);
    std::iota (std::begin (gblInds), std::end (gblInds), GO (0));
    const GO indexBase = 0;
    return Teuchos::rcp (new map_type (flag, gblInds.data (), lclNumInds, indexBase, comm));
  }

    Teuchos::RCP<const map_type>
    createSourceMap2 (const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
    {

        Teuchos::Array< GO > indexList;

        if(comm->getRank()==0){
            indexList.push_back(0);
        }
        if(comm->getRank()==1){
            indexList.push_back(0);
            indexList.push_back(1);
        }
        if(comm->getRank()==2){
            indexList.push_back(1);
        }
        if(comm->getRank()==3){
            indexList.push_back(0);
            indexList.push_back(2);
        }
        if(comm->getRank()==4){
            indexList.push_back(0);
            indexList.push_back(1);
            indexList.push_back(2);
            indexList.push_back(3);
        }
        if(comm->getRank()==5){
            indexList.push_back(1);
            indexList.push_back(3);
        }
        if(comm->getRank()==6){
            indexList.push_back(2);
        }
        if(comm->getRank()==7){
            indexList.push_back(2);
            indexList.push_back(3);
        }
        if(comm->getRank()==8){
            indexList.push_back(3);
        }
        const GO indexBase = 0;
        return Teuchos::rcp (new map_type (Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), indexList(), indexBase, comm));
    }
    
  // "Base value" should be some value, not default, not negative (in
  // case GO is unsigned, which is not default but possible).  Pick
  // baseValue() > comm->getSize().
  constexpr SC baseValue () {
    return static_cast<SC> (100.);
  }
  constexpr SC baseValue2 () {
        return static_cast<SC> (10.);
  }

  void
  fillTargetVector (vec_type& tgtVector)
  {
    tgtVector.putScalar (baseValue ());
  }

  void
  testVectorExport (bool& success,
                          Teuchos::FancyOStream& out,
                          vec_type& tgtVector,
                          vec_type& srcVector,
                          const export_type& exporter)
  {
    // Do some warm-up runs first that don't touch host.
    std::cout << "Testing..." << std::endl;
      tgtVector.putScalar (baseValue2 ());
      srcVector.putScalar (baseValue ());
      srcVector.describe(out,Teuchos::VERB_EXTREME);
      tgtVector.doExport (srcVector, exporter, Tpetra::INSERT);
      tgtVector.describe(out,Teuchos::VERB_EXTREME);
    // This run has the values about which we actually care.
//    fillTargetVector (tgtVector);
//    fillSourceVector (srcVector);
//    tgtVector.doExport (srcVector, exporter, Tpetra::INSERT);
//
//    tgtVector.sync_host ();
//    auto X_lcl_2d_h = tgtVector.getLocalViewHost ();
//    auto X_lcl_1d_h = Kokkos::subview (X_lcl_2d_h, Kokkos::ALL (), 0);
//
//    auto comm = tgtVector.getMap ()->getComm ();
//    const GO incrValue = GO (comm->getSize () - 1);
//    const GO expectedValue = baseValue () + incrValue;
//    TEST_EQUALITY( X_lcl_1d_h(0), expectedValue );
//
//    int lclSuccess = success ? 1 : 0;
//    int gblSuccess = 0; // output argument
//    using Teuchos::outArg;
//    using Teuchos::REDUCE_MIN;
//    using Teuchos::reduceAll;
//
//    reduceAll (*comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
//    TEST_ASSERT( gblSuccess == 1 );
  }

    void
    testMatrixExport (bool& success,
                      Teuchos::FancyOStream& out,
                      mat_type& tgtMat,
                      mat_type& srcMat,
                      const export_type& exporter)
    {
        // Do some warm-up runs first that don't touch host.
        std::cout << "Testing..." << std::endl;

        for (int i=0; i<srcMat.getMap()->getNodeNumElements (); i++) {

            std::vector<GO> indList(4);
            Teuchos::ArrayView<GO> indexList(indList);
            std::iota(indList.begin(),indList.end(),(GO) 0);
            Teuchos::Array<SC> valueList(4,10);
            srcMat.insertGlobalValues(srcMat.getMap()->getGlobalElement(i),indexList,valueList());
        }
        srcMat.fillComplete();
        srcMat.describe(out,Teuchos::VERB_EXTREME);
        tgtMat.doExport (srcMat, exporter, Tpetra::INSERT);
        tgtMat.fillComplete();
        tgtMat.describe(out,Teuchos::VERB_EXTREME);
        // This run has the values about which we actually care.
        //    fillTargetVector (tgtVector);
        //    fillSourceVector (srcVector);
        //    tgtVector.doExport (srcVector, exporter, Tpetra::INSERT);
        //
        //    tgtVector.sync_host ();
        //    auto X_lcl_2d_h = tgtVector.getLocalViewHost ();
        //    auto X_lcl_1d_h = Kokkos::subview (X_lcl_2d_h, Kokkos::ALL (), 0);
        //
        //    auto comm = tgtVector.getMap ()->getComm ();
        //    const GO incrValue = GO (comm->getSize () - 1);
        //    const GO expectedValue = baseValue () + incrValue;
        //    TEST_EQUALITY( X_lcl_1d_h(0), expectedValue );
        //
        //    int lclSuccess = success ? 1 : 0;
        //    int gblSuccess = 0; // output argument
        //    using Teuchos::outArg;
        //    using Teuchos::REDUCE_MIN;
        //    using Teuchos::reduceAll;
        //
        //    reduceAll (*comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
        //    TEST_ASSERT( gblSuccess == 1 );
    }
    
  TEUCHOS_UNIT_TEST( VectorExport2, MyIssue )
  {
    auto comm = Tpetra::getDefaultComm ();
    auto srcMap = createSourceMap2 (comm);
    auto tgtMap = createTargetMap2 (comm);
    std::cout << "in Test..." << std::endl;
    mat_type srcMat (srcMap,1);
    mat_type tgtMat (tgtMap,1);
    Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    srcMap->describe(*fancy,Teuchos::VERB_EXTREME);
    tgtMap->describe(*fancy,Teuchos::VERB_EXTREME);
    export_type exporter (srcMap, tgtMap);

//    testVectorExport (success, *fancy, tgtVector, srcVector,
//                            exporter);
    testMatrixExport (success, *fancy, tgtMat, srcMat, exporter);
  }

} // namespace (anonymous)

int
main (int argc, char* argv[])
{
  Tpetra::ScopeGuard tpetraScope (&argc, &argv);
    std::cout << "do something..." << std::endl;
  const int errCode =
    Teuchos::UnitTestRepository::runUnitTestsFromMain (argc, argv);
  return errCode;
}
