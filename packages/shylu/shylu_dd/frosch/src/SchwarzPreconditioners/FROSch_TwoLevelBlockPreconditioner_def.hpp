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
// Questions? Contact   Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//                      Christian Hochmuth (c.hochmuth@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_TWOLEVELBLOCKPRECONDITIONER_DEF_HPP
#define _FROSCH_TWOLEVELBLOCKPRECONDITIONER_DEF_HPP

#include <FROSch_TwoLevelBlockPreconditioner_decl.hpp>
using namespace Teuchos;
namespace FROSch {
    
    template <class SC,class LO,class GO,class NO>
    TwoLevelBlockPreconditioner<SC,LO,GO,NO>::TwoLevelBlockPreconditioner(CrsMatrixPtr k,
                                                                ParameterListPtr parameterList) :
    OneLevelPreconditioner<SC,LO,GO,NO> (k,parameterList),
    CoarseOperator_ ()
#ifdef FROSCH_TIMER
    ,SetupTwoLevel_(TimeMonitor_Type::getNewCounter("FROSch: TwoLevelBockPrec: Setup 1st & 2nd Level")),
    ComputeTwoLevel_(TimeMonitor_Type::getNewCounter("FROSch: TwoLevelBockPrec: Compute 1st & 2nd Level")),
    InitializeFirstLevel_(TimeMonitor_Type::getNewCounter("FROSch: TwoLevelBockPrec: Init 1st")),
    InitializeSecondLevel_(TimeMonitor_Type::getNewCounter("FROSch: TwoLevelBockPrec: Init 2nd Level")),
    ComputeFirstLevel_(TimeMonitor_Type::getNewCounter("FROSch: TwoLevelBockPrec: Compute 1st Level")),
    ComputeSecondLevel_(TimeMonitor_Type::getNewCounter("FROSch: TwoLevelBockPrec: Compute 2nd Level"))
#endif
    {

        if (this->ParameterList_->get("TwoLevel",true)) {            
            if (!this->ParameterList_->get("CoarseOperator Type","IPOUHarmonicCoarseOperator").compare("IPOUHarmonicCoarseOperator")) {
//                FROSCH_ASSERT(false,"not implemented for block.");
                this->ParameterList_->sublist("IPOUHarmonicCoarseOperator").sublist("InterfacePartitionOfUnity").set("Test Unconnected Interface",false);
                this->ParameterList_->sublist("IPOUHarmonicCoarseOperator").set("Mpi Ranks Coarse",parameterList->get("Mpi Ranks Coarse",0));
                this->ParameterList_->sublist("IPOUHarmonicCoarseOperator").set("Local problem ranks lower bound",parameterList->get("Local problem ranks lower bound",0));
                this->ParameterList_->sublist("IPOUHarmonicCoarseOperator").set("Local problem ranks upper bound",parameterList->get("Local problem ranks upper bound",this->MpiComm_->getSize()-1));
                this->ParameterList_->sublist("IPOUHarmonicCoarseOperator").set("Coarse problem ranks lower bound",parameterList->get("Coarse problem ranks lower bound",0));
                this->ParameterList_->sublist("IPOUHarmonicCoarseOperator").set("Coarse problem ranks upper bound",parameterList->get("Coarse problem ranks upper bound",this->MpiComm_->getSize()-1));
                this->ParameterList_->sublist("IPOUHarmonicCoarseOperator").set("Recycling",parameterList->get("Recycling","none"));
                CoarseOperator_ = IPOUHarmonicCoarseOperatorPtr(new IPOUHarmonicCoarseOperator<SC,LO,GO,NO>(k,sublist(parameterList,"IPOUHarmonicCoarseOperator")));
            } else if (!this->ParameterList_->get("CoarseOperator Type","IPOUHarmonicCoarseOperator").compare("GDSWCoarseOperator")) {
                this->ParameterList_->sublist("GDSWCoarseOperator").set("Test Unconnected Interface",false);
                this->ParameterList_->sublist("GDSWCoarseOperator").set("Mpi Ranks Coarse",parameterList->get("Mpi Ranks Coarse",0));
                this->ParameterList_->sublist("GDSWCoarseOperator").set("Local problem ranks lower bound",parameterList->get("Local problem ranks lower bound",0));
                this->ParameterList_->sublist("GDSWCoarseOperator").set("Local problem ranks upper bound",parameterList->get("Local problem ranks upper bound",this->MpiComm_->getSize()-1));
                this->ParameterList_->sublist("GDSWCoarseOperator").set("Coarse problem ranks lower bound",parameterList->get("Coarse problem ranks lower bound",0));
                this->ParameterList_->sublist("GDSWCoarseOperator").set("Coarse problem ranks upper bound",parameterList->get("Coarse problem ranks upper bound",this->MpiComm_->getSize()-1));
                this->ParameterList_->sublist("GDSWCoarseOperator").set("Recycling",parameterList->get("Recycling","none"));
                CoarseOperator_ = GDSWCoarseOperatorPtr(new GDSWCoarseOperator<SC,LO,GO,NO>(k,sublist(parameterList,"GDSWCoarseOperator")));
            } else if (!this->ParameterList_->get("CoarseOperator Type","IPOUHarmonicCoarseOperator").compare("RGDSWCoarseOperator")) {
                this->ParameterList_->sublist("RGDSWCoarseOperator").set("Test Unconnected Interface",false);
                this->ParameterList_->sublist("RGDSWCoarseOperator").set("Mpi Ranks Coarse",parameterList->get("Mpi Ranks Coarse",0));
                this->ParameterList_->sublist("RGDSWCoarseOperator").set("Local problem ranks lower bound",parameterList->get("Local problem ranks lower bound",0));
                this->ParameterList_->sublist("RGDSWCoarseOperator").set("Local problem ranks upper bound",parameterList->get("Local problem ranks upper bound",this->MpiComm_->getSize()-1));
                this->ParameterList_->sublist("RGDSWCoarseOperator").set("Coarse problem ranks lower bound",parameterList->get("Coarse problem ranks lower bound",0));
                this->ParameterList_->sublist("RGDSWCoarseOperator").set("Coarse problem ranks upper bound",parameterList->get("Coarse problem ranks upper bound",this->MpiComm_->getSize()-1));
                this->ParameterList_->sublist("RGDSWCoarseOperator").set("Recycling",parameterList->get("Recycling","none"));
                CoarseOperator_ = RGDSWCoarseOperatorPtr(new RGDSWCoarseOperator<SC,LO,GO,NO>(k,sublist(parameterList,"RGDSWCoarseOperator")));
            } else {
                FROSCH_ASSERT(false,"CoarseOperator Type unkown.");
            } // TODO: Add ability to disable individual levels

            this->LevelCombinationOperator_->addOperator(CoarseOperator_);
        }
    }
    
    template <class SC,class LO,class GO,class NO>
    int TwoLevelBlockPreconditioner<SC,LO,GO,NO>::initialize(UN dimension,
                                                             UNVecPtr dofsPerNodeVec,
                                                             DofOrderingVecPtr dofOrderingVec,
                                                             int overlap,
                                                             MapPtrVecPtr repeatedMapVec,
                                                             MultiVectorPtrVecPtr nullSpaceBasisVec,
                                                             MultiVectorPtrVecPtr nodeListVec,
                                                             MapPtrVecPtr2D dofsMapsVec,
                                                             GOVecPtr2D dirichletBoundaryDofsVec)
    {
        Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
        
        ////////////
        // Checks //
        ////////////
        UN nmbBlocks = dofsPerNodeVec.size();
        for (UN i = 0; i < dofOrderingVec.size(); i++ ) {
            DofOrdering dofOrdering = dofOrderingVec[i];
            FROSCH_ASSERT(dofOrdering == NodeWise || dofOrdering == DimensionWise || dofOrdering == Custom,"ERROR: Specify a valid DofOrdering.");
        }
        int ret = 0;
        //////////
        // Maps //
        //////////
        FROSCH_ASSERT(!repeatedMapVec.is_null(),"repeatedMapVec.is_null() = true. Please provide the repeated maps vector. The maps itself can be null and will be constructed.");
        for (UN i = 0; i < repeatedMapVec.size(); i++) {
            if (repeatedMapVec[i].is_null()) {
                FROSCH_ASSERT( i==0, "We can only construct a repeated map for a non block system");
                repeatedMapVec[i] = BuildRepeatedMap(this->K_); // Todo: Achtung, die UniqueMap könnte unsinnig verteilt sein. Falls es eine repeatedMap gibt, sollte dann die uniqueMap neu gebaut werden können. In diesem Fall, sollte man das aber basierend auf der repeatedNodesMap tun
            }
        }
        

        // Build dofsMaps and repeatedNodesMap
        MapPtrVecPtr repeatedNodesMapVec(dofsMapsVec.size());
        if (dofsMapsVec.is_null()) {
            if (0>BuildDofMapsVec(repeatedMapVec,dofsPerNodeVec,dofOrderingVec,repeatedNodesMapVec,dofsMapsVec)) ret -= 100; // Todo: Rückgabewerte
        } else {
            FROSCH_ASSERT(dofsMapsVec.size()==dofsPerNodeVec.size(),"dofsMapsVec.size()!=dofsPerNodeVec.size()");
            for (UN j=0; j<dofsMapsVec.size(); j++) {
                FROSCH_ASSERT(dofsMapsVec[j].size()==dofsPerNodeVec[j],"dofsMapsVec[block].size()!=dofsPerNodeVec[block]");
                for (UN i=0; i<dofsMapsVec[j].size(); i++) {
                    FROSCH_ASSERT(!dofsMapsVec[j][i].is_null(),"dofsMapsVec[block][i].is_null()");
                }
            }
            repeatedNodesMapVec = BuildNodeMapsFromDofMaps( dofsMapsVec, dofsPerNodeVec, dofOrderingVec );
        }
        
        
        //////////////////////////
        // Communicate nodeList //
        //////////////////////////
        if (!nodeListVec.is_null()) {
            for (UN i=0; i<nodeListVec.size(); i++) {
                if (!nodeListVec[i]->getMap()->isSameAs(*repeatedNodesMapVec[i])) {
                    Teuchos::RCP<Xpetra::MultiVector<SC,LO,GO,NO> > tmpNodeList = nodeListVec[i];
                    nodeListVec[i] = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(repeatedNodesMapVec[i],tmpNodeList->getNumVectors());
                    Teuchos::RCP<Xpetra::Import<LO,GO,NO> > scatter = Xpetra::ImportFactory<LO,GO,NO>::Build(tmpNodeList->getMap(),repeatedNodesMapVec[i]);
                    nodeListVec[i]->doImport(*tmpNodeList,*scatter,Xpetra::INSERT);
                }
            }
        }
        else{
            nodeListVec.resize(nmbBlocks);
        }
        
        
        //////////////////////////////////////////
        // Determine dirichletBoundaryDofs //
        //////////////////////////////////////////
        MapPtr repeatedMap;
        if (this->ParameterList_->get("Continuous Blocks",false))
            repeatedMap = MergeMapsCont( repeatedMapVec );
        else
            repeatedMap = MergeMaps( repeatedMapVec );
        
//        repeatedMap->describe(*fancy,Teuchos::VERB_EXTREME);
        
        if (dirichletBoundaryDofsVec.is_null()) {
            dirichletBoundaryDofsVec.resize(repeatedMapVec.size());
            LOVecPtr counterSub(repeatedMapVec.size(),0);
            for (UN j=0; j<dirichletBoundaryDofsVec.size(); j++) {
                dirichletBoundaryDofsVec[j] = GOVecPtr(repeatedMapVec[j]->getNodeNumElements());
            }
            GOVecPtr dirichletBoundaryDofs = FindOneEntryOnlyRowsGlobal(this->K_,repeatedMap);
            
            for (UN i=0; i<dirichletBoundaryDofs.size(); i++) {
                LO subNumber = -1;
                for (UN j = dofsMapsVec.size(); j > 0 ; j--) {
                    for (UN k=0; k<dofsMapsVec[j-1].size(); k++) {
                        if ( dirichletBoundaryDofs[i] <= dofsMapsVec[j-1][k]->getMaxAllGlobalIndex() ) {
                            subNumber = j-1;
                        }
                    }
                }
                dirichletBoundaryDofsVec[subNumber][counterSub[subNumber]] = dirichletBoundaryDofs[i];
                counterSub[subNumber]++;
            }
            for (UN i=0; i<dirichletBoundaryDofsVec.size(); i++) {
                dirichletBoundaryDofsVec[i].resize(counterSub[i]);
            }
            //dirichletBoundaryDofsVec = GOVecPtr2D(repeatedMapVec.size());
            
        }
        
#ifdef FROSCH_TIMER
        this->MpiComm_->barrier();
        TimeMonitor_Type SetupTwoLevelTM(*SetupTwoLevel_);
        TimeMonitor_Type InitializeFirstLevelTM(*InitializeFirstLevel_);
#endif

        ////////////////////////////////////
        // Initialize OverlappingOperator //
        ////////////////////////////////////
        if (!this->ParameterList_->get("OverlappingOperator Type","AlgebraicOverlappingOperator").compare("AlgebraicOverlappingOperator")) {
            AlgebraicOverlappingOperatorPtr algebraicOverlappigOperator = Teuchos::rcp_static_cast<AlgebraicOverlappingOperator<SC,LO,GO,NO> >(this->OverlappingOperator_);
            if (0>algebraicOverlappigOperator->initialize(overlap,repeatedMap)) ret -= 1;
        } else {
            FROSCH_ASSERT(false,"OverlappingOperator Type unkown.");
        }
        

        ///////////////////////////////
        // Initialize CoarseOperator //
        ///////////////////////////////
#ifdef FROSCH_TIMER
        this->MpiComm_->barrier();
        InitializeFirstLevelTM.~TimeMonitor();
        TimeMonitor_Type InitializeSecondLevelTM(*InitializeSecondLevel_);
        InitializeSecondLevelTM.setStackedTimer(Teuchos::null);
#endif
        if (this->ParameterList_->get("TwoLevel",true)) {
            if (!this->ParameterList_->get("CoarseOperator Type","IPOUHarmonicCoarseOperator").compare("IPOUHarmonicCoarseOperator")) {
                this->ParameterList_->sublist("IPOUHarmonicCoarseOperator").sublist("CoarseSolver").sublist("MueLu").set("Dimension",(int)dimension);

                MapPtrVecPtr repeatedMapWithOffsetVec = BuildMapsWithOffset( repeatedMapVec );
                // Build Null Space
                if (!this->ParameterList_->get("Null Space Type","Stokes").compare("Stokes")) {
                    nullSpaceBasisVec.resize(2);
                    nullSpaceBasisVec[0] = BuildNullSpace<SC,LO,GO,NO>(dimension,LaplaceNullSpace,repeatedMapWithOffsetVec[0],dofsPerNodeVec[0],dofsMapsVec[0], nodeListVec[0]);
                    nullSpaceBasisVec[1] = BuildNullSpace<SC,LO,GO,NO>(dimension,LaplaceNullSpace,repeatedMapWithOffsetVec[1],dofsPerNodeVec[1],dofsMapsVec[1], nodeListVec[1]);
                } else if (!this->ParameterList_->get("Null Space Type","Stokes").compare("LaplaceBlocks")) {
                    FROSCH_ASSERT(repeatedMapVec.size()==1,"Too many blocks for Null Space Type Laplace.");
                    nullSpaceBasisVec.resize( repeatedMapWithOffsetVec.size() );
                    for (int i=0; i<nullSpaceBasisVec.size(); i++) {
                        nullSpaceBasisVec[i] = BuildNullSpace<SC,LO,GO,NO>(dimension,LaplaceNullSpace,repeatedMapWithOffsetVec[i],dofsPerNodeVec[i],dofsMapsVec[i], nodeListVec[i]);
                    }
                } else if (!this->ParameterList_->get("Null Space Type","Stokes").compare("Input")) {
                    FROSCH_ASSERT(!nullSpaceBasisVec.is_null(),"Null Space Type is 'Input', but nullSpaceBasis.is_null().");
                } else {
                    FROSCH_ASSERT(false,"Null Space Type unknown.");
                }
                IPOUHarmonicCoarseOperatorPtr iPOUHarmonicCoarseOperator = Teuchos::rcp_static_cast<IPOUHarmonicCoarseOperator<SC,LO,GO,NO> >(CoarseOperator_);
                if (0>iPOUHarmonicCoarseOperator->initialize(dimension,dofsPerNodeVec,repeatedNodesMapVec,dofsMapsVec,nullSpaceBasisVec,nodeListVec,dirichletBoundaryDofsVec)) ret -=10;
            } else if (!this->ParameterList_->get("CoarseOperator Type","IPOUHarmonicCoarseOperator").compare("GDSWCoarseOperator")) {
                this->ParameterList_->sublist("GDSWCoarseOperator").sublist("CoarseSolver").sublist("MueLu").set("Dimension",(int)dimension);
                GDSWCoarseOperatorPtr gDSWCoarseOperator = Teuchos::rcp_static_cast<GDSWCoarseOperator<SC,LO,GO,NO> >(CoarseOperator_);
                if (0>gDSWCoarseOperator->initialize(dimension,dofsPerNodeVec,repeatedNodesMapVec,dofsMapsVec,dirichletBoundaryDofsVec,nodeListVec)) ret -=10;
            }
            else if (!this->ParameterList_->get("CoarseOperator Type","IPOUHarmonicCoarseOperator").compare("RGDSWCoarseOperator")) {
                this->ParameterList_->sublist("RGDSWCoarseOperator").sublist("CoarseSolver").sublist("MueLu").set("Dimension",(int)dimension);
                RGDSWCoarseOperatorPtr rGDSWCoarseOperator = Teuchos::rcp_static_cast<RGDSWCoarseOperator<SC,LO,GO,NO> >(CoarseOperator_);
                if (0>rGDSWCoarseOperator->initialize(dimension,dofsPerNodeVec,repeatedNodesMapVec,dofsMapsVec,dirichletBoundaryDofsVec,nodeListVec)) ret -=10;
            }
            else {
                FROSCH_ASSERT(false,"CoarseOperator Type unkown.");
            }
        }
#ifdef FROSCH_DETAIL_TIMER
        this->MpiComm_->barrier();
#endif
        this->IsInitialized_ = true;
        return ret;
    }
    
    template <class SC,class LO,class GO,class NO>
    int TwoLevelBlockPreconditioner<SC,LO,GO,NO>::compute()
    {
        int ret = 0;
#ifdef FROSCH_TIMER
#ifdef FROSCH_DETAIL_TIMER
        this->MpiComm_->barrier();
#endif
        TimeMonitor_Type SetupTwoLevelTM(*SetupTwoLevel_);
        TimeMonitor_Type ComputeTwoLevelTM(*ComputeTwoLevel_);
        TimeMonitor_Type ComputeSecondLevelTM(*ComputeSecondLevel_);
#endif
        if (this->ParameterList_->get("TwoLevel",true)) {
            if (0>CoarseOperator_->compute()) ret -= 10;
        }
#ifdef FROSCH_TIMER
        ComputeSecondLevelTM.~TimeMonitor();
        TimeMonitor_Type ComputeFirstLevelTM(*ComputeFirstLevel_);
#endif
        
        if (0>this->OverlappingOperator_->compute()) ret -= 1;
        
#ifdef FROSCH_DETAIL_TIMER
        this->MpiComm_->barrier();
#endif
        this->IsComputed_ = true;
        return ret;
    }
    
    template <class SC,class LO,class GO,class NO>
    void TwoLevelBlockPreconditioner<SC,LO,GO,NO>::describe(Teuchos::FancyOStream &out,
                                                   const Teuchos::EVerbosityLevel verbLevel) const
    {
        FROSCH_ASSERT(false,"describe() has be implemented properly...");
    }
    
    template <class SC,class LO,class GO,class NO>
    std::string TwoLevelBlockPreconditioner<SC,LO,GO,NO>::description() const
    {
        return "Two Level Block Preconditioner";
    }
    
    template <class SC,class LO,class GO,class NO>
    int TwoLevelBlockPreconditioner<SC,LO,GO,NO>::resetMatrix(CrsMatrixPtr &k)
    {
        this->K_ = k;
        this->OverlappingOperator_->resetMatrix(this->K_);
        if (this->ParameterList_->get("TwoLevel",true)) {
            CoarseOperator_->resetMatrix(this->K_);
            this->LevelCombinationOperator_->resetMatrix(this->K_);
        }
        return 0;
    }
    
    template <class SC,class LO,class GO,class NO>
    int TwoLevelBlockPreconditioner<SC,LO,GO,NO>::applyCoarseOperator(MultiVectorPtr &x,MultiVectorPtr &y)
    {
        this->LevelCombinationOperator_->applyCoarseOperator(*x,*y);
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    typename TwoLevelBlockPreconditioner<SC,LO,GO,NO>::CrsMatrixPtr TwoLevelBlockPreconditioner<SC,LO,GO,NO>::getPhi( )
    {
        
        FROSCH_ASSERT(this->ParameterList_->get("TwoLevel",true),"Can not getPhi() because a one-level method was used.")
        
        return CoarseOperator_->getPhi();
    }
    
}

#endif
