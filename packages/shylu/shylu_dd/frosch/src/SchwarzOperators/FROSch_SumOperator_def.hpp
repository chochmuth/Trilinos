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

#ifndef _FROSCH_SUMOPERATOR_DEF_HPP
#define _FROSCH_SUMOPERATOR_DEF_HPP

#include <FROSch_SumOperator_decl.hpp>

namespace FROSch {
    
    template <class SC,class LO,class GO,class NO>
    SumOperator<SC,LO,GO,NO>::SumOperator(CommPtr comm) :
    SchwarzOperator<SC,LO,GO,NO> (comm),
	OperatorVector_ (0),
	EnableOperators_ (0)
#ifdef FROSCH_TIMER
    ,ApplyTimer_(TimeMonitor_Type::getNewCounter("FROSch: Sum Operator: Apply"))
#endif
    {
        
    }
    
    template <class SC,class LO,class GO,class NO>
    SumOperator<SC,LO,GO,NO>::SumOperator(SchwarzOperatorPtrVecPtr operators) :
    SchwarzOperator<SC,LO,GO,NO> (operators.at(0)->getRangeMap()->getComm()),
	OperatorVector_ (0),
	EnableOperators_ (0)
#ifdef FROSCH_TIMER
    ,ApplyTimer_(TimeMonitor_Type::getNewCounter("FROSch: Sum Operator: Apply"))
#endif
    {
        OperatorVector_.push_back(operators.at(0));
        for (unsigned i=1; i<operators.size(); i++) {
            FROSCH_ASSERT(operators.at(i)->OperatorDomainMap().SameAs(OperatorVector_.at(0)->OperatorDomainMap()),"The DomainMaps of the operators are not identical.");
            FROSCH_ASSERT(operators.at(i)->OperatorRangeMap().SameAs(OperatorVector_.at(0)->OperatorRangeMap()),"The RangeMaps of the operators are not identical.");
            
            OperatorVector_.push_back(operators.at(i));
            EnableOperators_.push_back(true);
        }
    }
    
    template <class SC,class LO,class GO,class NO>
    SumOperator<SC,LO,GO,NO>::~SumOperator()
    {
        
    }
    
    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::initialize()
    {
        if (this->Verbose_) {
            std::cerr << "ERROR: Each of the Operators has to be initialized manually.";
        }
        return 0;
    }
    
    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::initialize(MapPtr repeatedMap)
    {
        if (this->Verbose_) {
            std::cerr << "ERROR: Each of the Operators has to be initialized manually.";
        }
        return 0;
    }
    
    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::compute()
    {
        if (this->Verbose_) {
            std::cerr << "ERROR: Each of the Operators has to be computed manually.";
        }
        return 0;
    }
    
    
    // Y = alpha * A^mode * X + beta * Y
    template <class SC,class LO,class GO,class NO>
    void SumOperator<SC,LO,GO,NO>::apply(const MultiVector &x,
                                         MultiVector &y,
                                         bool usePreconditionerOnly,
                                         Teuchos::ETransp mode,
                                         SC alpha,
                                         SC beta) const
    {
        
        
        if (OperatorVector_.size()>0) {
            if (OperatorVector_.size()==2 && OperatorVector_[OperatorVector_.size()-1]->getParameterList()->get("Mpi Ranks Coarse",0)>0) {
                
#ifdef FROSCH_TIMER
                TimeMonitor_Type ApplyTM(*ApplyTimer_);
#endif
                FROSCH_ASSERT(usePreconditionerOnly,"Parallel SumOperator is only implemented as a Preconditioner.");
                
                Teuchos::RCP<OverlappingOperator<SC,LO,GO,NO> > overlappingOp =
                    Teuchos::rcp_dynamic_cast<OverlappingOperator<SC,LO,GO,NO> >(OperatorVector_[0]);

                Teuchos::RCP<CoarseOperator<SC,LO,GO,NO> > coarseOp =
                    Teuchos::rcp_dynamic_cast<CoarseOperator<SC,LO,GO,NO> >(OperatorVector_[1]);
                
                FROSCH_ASSERT(overlappingOp->isComputed(),"Overlapping Operator is not computed.");
                FROSCH_ASSERT(coarseOp->isComputed(),"Coarse Operator is not computed.");
                
                MultiVectorPtr xTmp = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());
                *xTmp = x;
                
//                MapPtrVecPtr gatheringMaps = coarseOp->getGatheringMaps();
                MapPtr swapMap = coarseOp->getSwapMap();
                MapPtr overlappingMap = overlappingOp->getOverlappingMap();
                CrsMatrixPtr overlappingMatrix = overlappingOp->getOverlappingMatrix();
                SubdomainSolverPtr subdomainSolver = overlappingOp->getSubdomainSolver();
                ImporterPtr scatter = overlappingOp->getScatter();
                bool onFristLevelComm = overlappingOp->getOnLevelComm();
                
//                MultiVectorPtr xCoarseSolve = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(gatheringMaps[gatheringMaps.size()-1],x.getNumVectors());
//                MultiVectorPtr yCoarseSolve = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(gatheringMaps[gatheringMaps.size()-1],y.getNumVectors());
                MultiVectorPtr xCoarseSolve = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(swapMap,x.getNumVectors());
                MultiVectorPtr yCoarseSolve = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(swapMap,y.getNumVectors());
                coarseOp->applyPhiT(*xTmp,*xCoarseSolve);
                
                MultiVectorPtr xOverlap = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(overlappingMap,x.getNumVectors());
                MultiVectorPtr yOverlap = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(overlappingMap,x.getNumVectors());
                
                xOverlap->doImport(*xTmp,*scatter,Xpetra::INSERT);
                if (onFristLevelComm) {
                    yOverlap->replaceMap(overlappingMatrix->getRangeMap());
                    xOverlap->replaceMap(overlappingMatrix->getDomainMap());
                    subdomainSolver->apply(*xOverlap,*yOverlap,mode,1.0,0.0);
                }
                coarseOp->applyCoarseSolve(*xCoarseSolve,*yCoarseSolve,mode);
                
                xTmp->putScalar(0.0);
                yOverlap->replaceMap(overlappingMap);
                
                if (overlappingOp->getCombineMode() == Restricted){
                    GO globID = 0;
                    LO localID = 0;
                    for (unsigned i=0; i<y.getNumVectors(); i++) {
                        for (unsigned j=0; j<y.getMap()->getNodeNumElements(); j++) {
                            globID = y.getMap()->getGlobalElement(j);
                            localID = yOverlap->getMap()->getLocalElement(globID);
                            xTmp->getDataNonConst(i)[j] = yOverlap->getData(i)[localID];
                        }
                    }
                }
                else{
                    xTmp->doExport(*yOverlap,*scatter,Xpetra::ADD);
                }
                if (overlappingOp->getCombineMode() == Averaging) {
                    MultiVectorPtr multiplicity = overlappingOp->getMultiplicity();
                    ConstSCVecPtr scaling = multiplicity->getData(0);
                    for (unsigned j=0; j<xTmp->getNumVectors(); j++) {
                        SCVecPtr values = xTmp->getDataNonConst(j);
                        for (unsigned i=0; i<values.size(); i++) {
                            values[i] = values[i] / scaling[i];
                        }
                    }
                }
                
                y.update(alpha,*xTmp,beta);
                
                coarseOp->applyPhi(*yCoarseSolve,*xTmp);
                
                y.update(alpha,*xTmp,1.);

            }
            else{
                
                MultiVectorPtr xTmp = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());
                *xTmp = x; // Das brauche ich für den Fall das x=y
                UN itmp = 0;
                for (UN i=0; i<OperatorVector_.size(); i++) {
                    if (EnableOperators_[i]) {
                        OperatorVector_[i]->apply(*xTmp,y,usePreconditionerOnly,mode,alpha,beta);
                        if (itmp==0) beta = Teuchos::ScalarTraits<SC>::one();
                        itmp++;
                    }
                }
            }
        } else {
            y.update(alpha,x,beta);
        }
    }
    
    template <class SC,class LO,class GO,class NO>
    typename SumOperator<SC,LO,GO,NO>::ConstMapPtr SumOperator<SC,LO,GO,NO>::getDomainMap() const
    {
        return OperatorVector_[0]->getDomainMap();
    }
    
    template <class SC,class LO,class GO,class NO>
    typename SumOperator<SC,LO,GO,NO>::ConstMapPtr SumOperator<SC,LO,GO,NO>::getRangeMap() const
    {
        return OperatorVector_[0]->getRangeMap();
    }
    
    template <class SC,class LO,class GO,class NO>
    void SumOperator<SC,LO,GO,NO>::describe(Teuchos::FancyOStream &out,
                                            const Teuchos::EVerbosityLevel verbLevel) const
    {
        FROSCH_ASSERT(0!=0,"describe() has be implemented properly...");
    }
    
    template <class SC,class LO,class GO,class NO>
    std::string SumOperator<SC,LO,GO,NO>::description() const
    {
        std::string labelString = "Sum operator: ";
        
        for (UN i=0; i<OperatorVector_.size(); i++) {
            labelString += OperatorVector_.at(i)->description();
            if (i<OperatorVector_.size()-1) {
                labelString += ",";
            }
        }
        return labelString;
    }
    
    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)
    {
        int ret = 0;
        if (OperatorVector_.size()>0) {
            if (!op->getDomainMap()->isSameAs(*OperatorVector_[0]->getDomainMap())) {
                if (this->Verbose_) std::cerr << "SumOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getDomainMap().isSameAs(OperatorVector_[0]->getDomainMap())\n";
                ret -= 1;
            }
            if (!op->getRangeMap()->isSameAs(*OperatorVector_[0]->getRangeMap())){
                if (this->Verbose_) std::cerr << "SumOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getRangeMap().isSameAs(OperatorVector_[0]->getRangeMap())\n";
                ret -= 10;
            }
            //FROSCH_ASSERT(op->OperatorDomainMap().SameAs(OperatorVector_.at(0)->OperatorDomainMap()),"The DomainMaps of the operators are not identical.");
            //FROSCH_ASSERT(op->OperatorRangeMap().SameAs(OperatorVector_.at(0)->OperatorRangeMap()),"The RangeMaps of the operators are not identical.");
        }
        OperatorVector_.push_back(op);
        EnableOperators_.push_back(true);
        return ret;
    }

    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::addOperators(SchwarzOperatorPtrVecPtr operators)
    {
        int ret = 0;
        for (UN i=1; i<operators.size(); i++) {
            if (0>addOperator(operators[i])) ret -= pow(10,i);
        }
        return ret;
    }

    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::resetOperator(UN iD,
    											SchwarzOperatorPtr op)
    {
        FROSCH_ASSERT(iD<OperatorVector_.size(),"iD exceeds the length of the OperatorVector_");
        int ret = 0;
        if (!op->getDomainMap().isSameAs(OperatorVector_[0]->getDomainMap())) {
            if (this->Verbose_) std::cerr << "SumOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getDomainMap().isSameAs(OperatorVector_[0]->getDomainMap())\n";
            ret -= 1;
        }
        if (!op->getRangeMap().isSameAs(OperatorVector_[0]->getRangeMap())){
            if (this->Verbose_) std::cerr << "SumOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getRangeMap().isSameAs(OperatorVector_[0]->getRangeMap())\n";
            ret -= 10;
        }
        OperatorVector_[iD] = op;
        return ret;
    }

    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::enableOperator(UN iD,
    											 bool enable)
	{
    	EnableOperators_[iD] = enable;
    	return 0;
	}

    template <class SC,class LO,class GO,class NO>
    typename SumOperator<SC,LO,GO,NO>::UN SumOperator<SC,LO,GO,NO>::getNumOperators()
    {
    	return OperatorVector_.size();
    }
}

#endif
