// @HEADER
//
// ***********************************************************************
//
//             Xpetra: A linear algebra interface package
//                  Copyright 2012 Sandia Corporation
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
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#include <iostream>
#include <numeric>
#include <vector>


#include <Teuchos_RCP.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

#include <Xpetra_Map.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_Vector.hpp>
#include <Xpetra_MultiVector.hpp>
#include <Xpetra_Export.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Xpetra_CrsMatrixFactory.hpp>
#include <Xpetra_ExportFactory.hpp>
#ifdef HAVE_XPETRA_TPETRA

#include <TpetraCore_config.h>

#if ((defined(HAVE_TPETRA_INST_OPENMP) || defined(HAVE_TPETRA_INST_SERIAL)) && \
    (defined(HAVE_TPETRA_INST_INT_INT) || defined(HAVE_TPETRA_INST_INT_LONG_LONG)) && \
    defined(HAVE_TPETRA_INST_DOUBLE))

// Choose types Tpetra is instantiated on
typedef double Scalar;
typedef int    LocalOrdinal;
typedef long long GlobalOrdinal;
//#if defined(HAVE_TPETRA_INST_INT_INT)
//typedef int    GlobalOrdinal;
//#elif defined(HAVE_TPETRA_INST_INT_LONG_LONG)
//typedef long long GlobalOrdinal;
//#endif
//#if defined(HAVE_TPETRA_INST_OPENMP)
//typedef Kokkos::Compat::KokkosOpenMPWrapperNode Node;
//#elif defined(HAVE_TPETRA_INST_SERIAL)
//typedef Kokkos::Compat::KokkosSerialWrapperNode Node;
//#endif
typedef Kokkos::Compat::KokkosSerialWrapperNode Node;
int main(int argc, char *argv[]) {

  using Teuchos::RCP;
  using Teuchos::rcp;

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);

  bool success = false;
  bool verbose = false;
  try {
    RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();
      Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

    Xpetra::UnderlyingLib lib = Xpetra::UseTpetra;

    RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > mapSrc;
    RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > mapTgt;
    {
          Teuchos::Array< GlobalOrdinal > indexList;
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
          const GlobalOrdinal indexBase = 0;
          mapSrc = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(lib, -1, indexList(), indexBase, comm);
      }
     
      {
          Teuchos::Array<  GlobalOrdinal > indexList;
          LocalOrdinal numMyRows = 0;
          if(comm->getRank()==0){
              indexList.push_back(0);
              indexList.push_back(1);
              numMyRows=2;
          }
          if(comm->getRank()==1){
              
          }
          if(comm->getRank()==2){
              
          }
          if(comm->getRank()==3){
              indexList.push_back(2);
              numMyRows=1;
          }
          if(comm->getRank()==4){
              
          }
          if(comm->getRank()==5){
              
          }
          if(comm->getRank()==6){
              indexList.push_back(3);
              numMyRows=1;
          }
          if(comm->getRank()==7){
              
          }
          if(comm->getRank()==8){
              
          }
          const GlobalOrdinal indexBase = 0;
          
//          mapTgt = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(lib, -1, indexList(), indexBase, comm);
          mapTgt = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(lib, -1, numMyRows, indexBase, comm);
          mapTgt->describe(*fancy,Teuchos::VERB_EXTREME);
      }
    const size_t numMyElements = mapSrc->getNodeNumElements();
    RCP<Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > srcA =  Xpetra::CrsMatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(mapSrc, 4);
      for (size_t i=0; i<numMyElements; i++) {
          
          std::vector<GlobalOrdinal> indList(4);
          Teuchos::ArrayView<GlobalOrdinal> indexList(indList);
          std::iota(indList.begin(),indList.end(),(GlobalOrdinal) 0);
          Teuchos::Array<Scalar> valueList(4,10);
          srcA->insertGlobalValues(srcA->getMap()->getGlobalElement(i),indexList,valueList());
      }

    srcA->fillComplete();
      
      srcA->describe(*fancy,Teuchos::VERB_EXTREME);

    RCP<Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > tgtA =  Xpetra::CrsMatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(mapTgt, 4);
      
      RCP<Xpetra::Export<LocalOrdinal, GlobalOrdinal, Node> > exporter = Xpetra::ExportFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(mapSrc,mapTgt);
      
      tgtA->doExport(*srcA,*exporter,Xpetra::INSERT);
      comm->barrier();      comm->barrier();      comm->barrier();
            std::cout << "PRE FILL COMPLETE TARGET"<< std::endl;
      tgtA->fillComplete();
      
        tgtA->describe(*fancy,Teuchos::VERB_EXTREME);
    success = true;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}
#else
int main(int argc, char *argv[]) { std::cout << "Tpetra is not instantiated on SC=double, GO=int/long long and Node=Serial/OpenMP. Skip example." << std::endl; return EXIT_SUCCESS; }
#endif // Tpetra instantiated on SC=double, GO=int/long long and Node=Serial/OpenMP
#else
int main(int argc, char *argv[]) { std::cout << "Xpetra has been compiled without Tpetra support. Skip example." << std::endl; return EXIT_SUCCESS; }
#endif // HAVE_XPETRA_TPETRA
