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

#include <Tpetra_Core.hpp>

// Xpetra include
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Xpetra_EpetraCrsMatrix.hpp>
#endif
#include <Xpetra_Parameters.hpp>
#include <Xpetra_UseDefaultTypes.hpp>

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
    
    int NumberNodeElements = 4;
    My_CLP.setOption("NNE",&NumberNodeElements,"NumberNodeElements");
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
    
    
    UN NumberOfBlocks = 3;
    
    UnderlyingLib xpetraLib = UseTpetra;
    if (useepetra) {
        xpetraLib = UseEpetra;
    } else {
        xpetraLib = UseTpetra;
    }
    int color = 0;
    RCP<const Comm<int> > Comm = CommWorld->split(color,CommWorld->getRank());

//    if(Comm->getRank()==0) {
//        cout << "##################\n# Parameter List #\n##################" << endl;
//        parameterList->print(cout);
//        cout << endl;
//    }
    
    if (color==0) {
        
        RCP<ParameterList> parameterList = getParametersFromXmlFile(xmlFile);
        

        
        ArrayRCP<UN> dofsPerNodeVector(NumberOfBlocks);
        ArrayRCP<DofOrdering> dofsOrderingVector(NumberOfBlocks);
        dofsPerNodeVector[0] = 2; dofsPerNodeVector[1] = 1; dofsPerNodeVector[2] = 2;// dofsPerNodeVector[3] = 2;
        dofsOrderingVector[0] = NodeWise; dofsOrderingVector[1] = NodeWise; dofsOrderingVector[2] = DimensionWise; //dofsOrderingVector[3] = DimensionWise;
        UN TotalDofs = 0;
        for (UN i=0; i<dofsPerNodeVector.size(); i++)
            TotalDofs += dofsPerNodeVector[i];
        
        Array<Array< GO > > indicesVec ( TotalDofs, Array< GO > (NumberNodeElements));
        
        
        ArrayRCP<ArrayRCP<RCP<Map<LO,GO,NO> > > > DofMapsVec(NumberOfBlocks);
        for (UN i=0; i<NumberOfBlocks; i++) {
            DofMapsVec[i].resize(dofsPerNodeVector[i]);
        }
        
        //Build first three dof maps; NodeWise
        for (UN i=0; i<dofsPerNodeVector[0] + dofsPerNodeVector[1] ; i++) {
            for (UN j=0; j<(UN) NumberNodeElements; j++) {
                indicesVec[i][j] = j*(dofsPerNodeVector[0] + dofsPerNodeVector[1]) + i + (dofsPerNodeVector[0] + dofsPerNodeVector[1]) * NumberNodeElements * Comm->getRank();
            }
        }
        UN offsetDofs = dofsPerNodeVector[0] + dofsPerNodeVector[1];
        GO offset = offsetDofs * NumberNodeElements * Comm->getSize();
        //Build next four dof maps; DimensionWise
        for (UN i=0; i<dofsPerNodeVector[2] ; i++) {
            for (UN j=0; j<(UN) NumberNodeElements; j++) {
                indicesVec[i+offsetDofs][j] = i*Comm->getSize()*NumberNodeElements + j + offset + NumberNodeElements*Comm->getRank();
            }
        }
        
        RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout));
        UN counter = 0;
        for (UN i=0; i<NumberOfBlocks; i++) {
            for (UN j=0; j<dofsPerNodeVector[i]; j++) {
                DofMapsVec[i][j] = MapFactory<LO,GO,NO>::Build(xpetraLib,-1,indicesVec[counter](),0,Comm);
                DofMapsVec[i][j]->describe(*fancy,Teuchos::VERB_EXTREME);
                counter++;
            }
        }
        
        ArrayRCP<RCP<Map<LO,GO,NO> > > NodesMapVec = BuildNodeMapsFromDofMaps( DofMapsVec, dofsPerNodeVector, dofsOrderingVector );
        Comm->barrier(); if (Comm->getRank()==0) cout << "\n#############\n# Node Maps: #\n#############" << endl;
        for (UN i=0; i<NodesMapVec.size(); i++) {
            NodesMapVec[i]->describe(*fancy,Teuchos::VERB_EXTREME);
        }

        Comm->barrier(); if (Comm->getRank()==0) cout << "\n#############\n# Finished! #\n#############" << endl;
    }
    
    return(EXIT_SUCCESS);
    
}
