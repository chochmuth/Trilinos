//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
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
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#include <ShyLU_DDFROSch_config.h>

#include <mpi.h>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>

// Galeri::Xpetra
#include "Galeri_XpetraProblemFactory.hpp"
#include "Galeri_XpetraMatrixTypes.hpp"
#include "Galeri_XpetraParameters.hpp"
#include "Galeri_XpetraUtils.hpp"
#include "Galeri_XpetraMaps.hpp"


#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Epetra_MpiComm.h>
#endif

#include <Tpetra_Core.hpp>

// Xpetra include
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Xpetra_EpetraCrsMatrix.hpp>
#endif
#include <Xpetra_Parameters.hpp>
#include <Xpetra_UseDefaultTypes.hpp>

// FROSCH thyra includes
#include <FROSch_TwoLevelBlockPreconditioner_def.hpp>
#include <FROSch_Tools_def.hpp>

typedef unsigned                                    UN;
typedef Scalar                                      SC;
typedef LocalOrdinal                                LO;
typedef GlobalOrdinal                               GO;
typedef KokkosClassic::DefaultNode::DefaultNodeType NO;

using namespace std;
using namespace Teuchos;
using namespace Xpetra;
using namespace FROSch;

int main(int argc, char *argv[])
{
    oblackholestream blackhole;
    GlobalMPISession mpiSession(&argc,&argv,&blackhole);
    
    RCP<const Comm<int> > CommWorld = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();
    
    CommandLineProcessor My_CLP;
    
    RCP<FancyOStream> out = VerboseObjectBase::getDefaultOStream();
    
    int M = 4;
    My_CLP.setOption("M",&M,"H / h.");
    int Dimension = 2;
    My_CLP.setOption("DIM",&Dimension,"Dimension.");
    int NumberOfBlocks = 2;
    My_CLP.setOption("NB",&NumberOfBlocks,"Number of blocks.");
    int DofsPerNode = 1;
    My_CLP.setOption("DPN",&DofsPerNode,"Dofs per node.");
    int DOFOrdering = 0;
    My_CLP.setOption("ORD",&DOFOrdering,"Dofs ordering (NodeWise=0, DimensionWise=1, Custom=2).");
    string xmlFile = "ParameterList.xml";
    My_CLP.setOption("PLIST",&xmlFile,"File name of the parameter list.");
    bool useepetra = false;
    My_CLP.setOption("USEEPETRA","USETPETRA",&useepetra,"Use Epetra infrastructure for the linear algebra.");
    
    My_CLP.recogniseAllOptions(true);
    My_CLP.throwExceptions(false);
    CommandLineProcessor::EParseCommandLineReturn parseReturn = My_CLP.parse(argc,argv);
    if(parseReturn == CommandLineProcessor::PARSE_HELP_PRINTED) {
        return(EXIT_SUCCESS);
    }
    
    int N;
    int color=1;
    if (Dimension == 2) {
        N = (int) (pow(CommWorld->getSize(),1/2.) + 100*numeric_limits<double>::epsilon()); // 1/H
        if (CommWorld->getRank()<N*N) {
            color=0;
        }
    } else if (Dimension == 3) {
        N = (int) (pow(CommWorld->getSize(),1/3.) + 100*numeric_limits<double>::epsilon()); // 1/H
        if (CommWorld->getRank()<N*N*N) {
            color=0;
        }
    } else {
        assert(false);
    }
    
    UnderlyingLib xpetraLib = UseTpetra;
    if (useepetra) {
        xpetraLib = UseEpetra;
    } else {
        xpetraLib = UseTpetra;
    }
    
    RCP<const Comm<int> > Comm = CommWorld->split(color,CommWorld->getRank());
    
    if (color==0) {
        
        RCP<ParameterList> parameterList = getParametersFromXmlFile(xmlFile);
        
        ArrayRCP<RCP<Matrix<SC,LO,GO,NO> > > K(NumberOfBlocks);
        ArrayRCP<RCP<Map<LO,GO,NO> > > RepeatedMaps(NumberOfBlocks);
        ArrayRCP<RCP<MultiVector<SC,LO,GO,NO> > > Coordinates(NumberOfBlocks);
        ArrayRCP<UN> dofsPerNodeVector(NumberOfBlocks);
        ArrayRCP<DofOrdering> dofOrderingsVector(NumberOfBlocks);
        

        
        for (UN block=0; block<(UN) NumberOfBlocks; block++) {
            Comm->barrier(); if (Comm->getRank()==0) cout << "###################\n# Assembly Block " << block << " #\n###################\n" << endl;
            
            dofsPerNodeVector[block] = (UN) DofsPerNode;//max(int(DofsPerNode-block),1);
            std::cout << "dofsPerNodeVector[block]:" << dofsPerNodeVector[block] << std::endl;
            if (DOFOrdering==0) dofOrderingsVector[block] = NodeWise;
            else if (DOFOrdering==1) dofOrderingsVector[block] = DimensionWise;
            else dofOrderingsVector[block] = Custom;

            ParameterList GaleriList;
            GaleriList.set("nx", int(N*(M)));
            GaleriList.set("ny", int(N*(M)));
            GaleriList.set("nz", int(N*(M)));
            GaleriList.set("mx", int(N));
            GaleriList.set("my", int(N));
            GaleriList.set("mz", int(N));
            
            RCP<const Map<LO,GO,NO> > UniqueMapTmp;
            RCP<MultiVector<SC,LO,GO,NO> > CoordinatesTmp;
            RCP<Matrix<SC,LO,GO,NO> > KTmp;
            if (Dimension==2) {
                UniqueMapTmp = Galeri::Xpetra::CreateMap<LO,GO,NO>(xpetraLib,"Cartesian2D",Comm,GaleriList); // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); nodeMap->describe(*fancy,VERB_EXTREME);
                CoordinatesTmp = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map<LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("2D",UniqueMapTmp,GaleriList);
                RCP<Galeri::Xpetra::Problem<Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> > > Problem = Galeri::Xpetra::BuildProblem<SC,LO,GO,Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("Laplace2D",UniqueMapTmp,GaleriList);
                KTmp = Problem->BuildMatrix();
            } else if (Dimension==3) {
                UniqueMapTmp = Galeri::Xpetra::CreateMap<LO,GO,NO>(xpetraLib,"Cartesian3D",Comm,GaleriList); // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); nodeMap->describe(*fancy,VERB_EXTREME);
                CoordinatesTmp = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map<LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("3D",UniqueMapTmp,GaleriList);
                RCP<Galeri::Xpetra::Problem<Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> > > Problem = Galeri::Xpetra::BuildProblem<SC,LO,GO,Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("Laplace3D",UniqueMapTmp,GaleriList);
                KTmp = Problem->BuildMatrix();
            }
            
            RCP<Map<LO,GO,NO> > UniqueMap;
            
            if (DOFOrdering == 0) {
                Array<GO> uniqueMapArray(dofsPerNodeVector[block]*UniqueMapTmp->getNodeNumElements());
                for (LO i=0; i<(LO) UniqueMapTmp->getNodeNumElements(); i++) {
                    for (UN j=0; j<dofsPerNodeVector[block]; j++) {
                        uniqueMapArray[dofsPerNodeVector[block]*i+j] = dofsPerNodeVector[block]*UniqueMapTmp->getGlobalElement(i)+j;
                    }
                }
                UniqueMap = MapFactory<LO,GO,NO>::Build(xpetraLib,-1,uniqueMapArray(),0,Comm);
                K[block] = MatrixFactory<SC,LO,GO,NO>::Build(UniqueMap,KTmp->getGlobalMaxNumRowEntries());
                for (LO i=0; i<(LO) UniqueMapTmp->getNodeNumElements(); i++) {
                    ArrayView<const LO> indices;
                    ArrayView<const SC> values;
                    KTmp->getLocalRowView(i,indices,values);
                    
                    for (UN j=0; j<dofsPerNodeVector[block]; j++) {
                        Array<GO> indicesArray(indices.size());
                        for (LO k=0; k<indices.size(); k++) {
                            indicesArray[k] = dofsPerNodeVector[block]*KTmp->getColMap()->getGlobalElement(indices[k])+j;
                        }
                        K[block]->insertGlobalValues(dofsPerNodeVector[block]*KTmp->getRowMap()->getGlobalElement(i)+j,indicesArray(),values);
                    }
                }
                K[block]->fillComplete();
            } else if (DOFOrdering == 1) {
                Array<GO> uniqueMapArray(dofsPerNodeVector[block]*UniqueMapTmp->getNodeNumElements());
                for (LO i=0; i<(LO) UniqueMapTmp->getNodeNumElements(); i++) {
                    for (UN j=0; j<dofsPerNodeVector[block]; j++) {
                        uniqueMapArray[i+UniqueMapTmp->getNodeNumElements()*j] = UniqueMapTmp->getGlobalElement(i)+(UniqueMapTmp->getMaxAllGlobalIndex()+1)*j;
                    }
                }
                
                UniqueMap = MapFactory<LO,GO,NO>::Build(xpetraLib,-1,uniqueMapArray(),0,Comm);
                K[block] = MatrixFactory<SC,LO,GO,NO>::Build(UniqueMap,KTmp->getGlobalMaxNumRowEntries());
                for (LO i=0; i<(LO) UniqueMapTmp->getNodeNumElements(); i++) {
                    ArrayView<const LO> indices;
                    ArrayView<const SC> values;
                    KTmp->getLocalRowView(i,indices,values);
                    
                    for (UN j=0; j<dofsPerNodeVector[block]; j++) {
                        Array<GO> indicesArray(indices.size());
                        for (LO k=0; k<indices.size(); k++) {
                            indicesArray[k] = KTmp->getColMap()->getGlobalElement(indices[k])+(KTmp->getColMap()->getMaxAllGlobalIndex()+1)*j;
                        }
                        K[block]->insertGlobalValues(UniqueMapTmp->getGlobalElement(i)+(UniqueMapTmp->getMaxAllGlobalIndex()+1)*j,indicesArray(),values);
                    }
                }
                K[block]->fillComplete();
            } else if (DOFOrdering == 2) {
                assert(false); // TODO: Andere Sortierung implementieren
            } else {
                assert(false);
            }
            
            RepeatedMaps[block] = FROSch::BuildRepeatedMap<SC,LO,GO,NO>(K[block]);
            //RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); RepeatedMaps[block]->describe(*fancy,VERB_EXTREME);
        }
        ArrayRCP<RCP<Map<LO,GO,NO> > > NodesMaps;
        ArrayRCP<ArrayRCP<RCP<Map<LO,GO,NO> > > > DofMaps;
        std::cout << "RepeatedMaps"<<std::endl;
        RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout));
        for (int i=0; i<RepeatedMaps.size(); i++) {
            RepeatedMaps[i]->describe(*fancy,VERB_EXTREME);
//            K[i]->describe(*fancy,VERB_EXTREME);
        }
        Comm->barrier();        Comm->barrier();        Comm->barrier();
        BuildDofMapsVec(RepeatedMaps,
                        dofsPerNodeVector,
                        dofOrderingsVector,
                        NodesMaps,
                        DofMaps);
        
        std::cout << "DofMaps:"<<DofMaps.size() << std::endl;
        for (int i=0; i<DofMaps.size(); i++) {
            std::cout << "size "<<i <<":"  << DofMaps[i].size()<< " DofVector:"<<dofsPerNodeVector[i] <<  std::endl;
            Comm->barrier();        Comm->barrier();        Comm->barrier();
            for (int j=0; j<DofMaps[i].size(); j++) {
                DofMaps[i][j]->describe(*fancy,VERB_EXTREME);
            }
        }

        
        
        Comm->barrier(); if (Comm->getRank()==0) cout << "##############################\n# Assembly Monolithic System #\n##############################\n" << endl;
        
        RCP<Matrix<SC,LO,GO,NO> > KMonolithic;
        if (NumberOfBlocks>1) {
            
            Array<GO> uniqueMapArray(0);
            GO tmpOffset = 0;
            for (UN block=0; block<(UN) NumberOfBlocks; block++) {
                ArrayView<const GO> tmpgetGlobalElements = K[block]->getMap()->getNodeElementList();
                for (LO i=0; i<tmpgetGlobalElements.size(); i++) {
                    uniqueMapArray.push_back(tmpgetGlobalElements[i]+tmpOffset);
                }
                tmpOffset += K[block]->getMap()->getMaxAllGlobalIndex()+1;
            }
            RCP<Map<LO,GO,NO> > UniqueMapMonolithic = MapFactory<LO,GO,NO>::Build(xpetraLib,-1,uniqueMapArray(),0,Comm);
            
            tmpOffset = 0;
            KMonolithic = MatrixFactory<SC,LO,GO,NO>::Build(UniqueMapMonolithic,K[0]->getGlobalMaxNumRowEntries());
            for (UN block=0; block<(UN) NumberOfBlocks; block++) {
                for (LO i=0; i<(LO) K[block]->getNodeNumRows(); i++) {
                    ArrayView<const LO> indices;
                    ArrayView<const SC> values;
                    K[block]->getLocalRowView(i,indices,values);
                    Array<GO> indicesGlobal(indices.size());
                    for (UN j=0; j<indices.size(); j++) {
                        indicesGlobal[j] = K[block]->getColMap()->getGlobalElement(indices[j])+tmpOffset;
                    }
                    KMonolithic->insertGlobalValues(K[block]->getMap()->getGlobalElement(i)+tmpOffset,indicesGlobal(),values);
                }
                tmpOffset += K[block]->getMap()->getMaxAllGlobalIndex()+1;
            }
            KMonolithic->fillComplete();
        } else if (NumberOfBlocks==1) {
            KMonolithic = K[0];
        } else {
            assert(false);
        }
       
        
        ArrayRCP< RCP<MultiVector<SC,LO,GO,NO> > > nodesDummy = Teuchos::null;
        ArrayRCP< RCP<MultiVector<SC,LO,GO,NO> > > nullSpaceDummy = Teuchos::null;
        
        RCP<TwoLevelBlockPreconditioner<SC,LO,GO,NO> > TLBP(new TwoLevelBlockPreconditioner<SC,LO,GO,NO>(KMonolithic,parameterList));
        
        TLBP->initialize(Dimension,dofsPerNodeVector,dofOrderingsVector,parameterList->get("Overlap",0),RepeatedMaps,nullSpaceDummy,nodesDummy,DofMaps);
        
        TLBP->compute();
        
        Comm->barrier();
        if(Comm->getRank()==0) {
            cout << "##################\n# Parameter List #\n##################" << endl;
            parameterList->print(cout);
            cout << endl;
        }
        
        Comm->barrier(); if (Comm->getRank()==0) cout << "\n#############\n# Finished! #\n#############" << endl;
    }
    
    return(EXIT_SUCCESS);
    
}
