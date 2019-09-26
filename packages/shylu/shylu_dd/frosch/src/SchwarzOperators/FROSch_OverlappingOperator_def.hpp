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

#ifndef _FROSCH_OVERLAPPINGOPERATOR_DEF_HPP
#define _FROSCH_OVERLAPPINGOPERATOR_DEF_HPP

#include <FROSch_OverlappingOperator_decl.hpp>

namespace FROSch {
    
    template <class SC,class LO,class GO,class NO>
    OverlappingOperator<SC,LO,GO,NO>::OverlappingOperator(CrsMatrixPtr k,
                                                          ParameterListPtr parameterList) :
    SchwarzOperator<SC,LO,GO,NO> (k,parameterList),
    OverlappingMatrix_ (),
    OverlappingMap_ (),
    Scatter_(),
    GatherRestricted_(),
    SubdomainSolver_ (),
    Multiplicity_(),
    Combine_(),
    LevelID_(this->ParameterList_->get("Level ID",1)),
    OnFirstLevelComm_(false),
    FirstLevelSolveComm_(),
    XOverlapEpetra_()
#ifdef FROSCH_TIMER
    ,OverlapTimer_(TimeMonitor_Type::getNewCounter("FROSch: Overlapping Operator("+ Teuchos::toString(LevelID_)+"): Build Overlap")),
    ExtractTimer_(TimeMonitor_Type::getNewCounter("FROSch: Overlapping Operator("+ Teuchos::toString(LevelID_)+"): Extract Local Matrices")),
    ComputeTimer_(TimeMonitor_Type::getNewCounter("FROSch: Overlapping Operator("+ Teuchos::toString(LevelID_)+"): Compute")),
    FullSetupTimer_(TimeMonitor_Type::getNewCounter("FROSch: Overlapping Operator("+ Teuchos::toString(LevelID_)+"): Full Setup")),
    ApplyTimer_(TimeMonitor_Type::getNewCounter("FROSch: Overlapping Operator("+ Teuchos::toString(LevelID_)+"): Apply")),
    SymbolicFacTimer_(TimeMonitor_Type::getNewCounter("FROSch: Overlapping Operator("+ Teuchos::toString(LevelID_)+"): Compute: Symbolic Factorization")),
    NumericFacTimer_(TimeMonitor_Type::getNewCounter("FROSch: Overlapping Operator("+ Teuchos::toString(LevelID_)+"): Compute: Numeric Factorization"))
#endif
#ifdef FROSCH_DETAIL_TIMER
    ,ApplyRestTimer_(TimeMonitor_Type::getNewCounter("FROSch: Overlapping Operator("+ Teuchos::toString(LevelID_)+"): Apply: Gather restriction")),
    ApplyNormalTimer_(TimeMonitor_Type::getNewCounter("FROSch: Overlapping Operator("+ Teuchos::toString(LevelID_)+"): Apply: Gather normal")),
    ApplyScatterTimer_(TimeMonitor_Type::getNewCounter("FROSch: Overlapping Operator("+ Teuchos::toString(LevelID_)+"): Apply: Scatter")),
    ApplySolveTimer_(TimeMonitor_Type::getNewCounter("FROSch: Overlapping Operator("+ Teuchos::toString(LevelID_)+"): Apply: Solve"))
#endif
    {

        if (!this->ParameterList_->get("Overlapping Operator Combination","Restricted").compare("Averaging")) {
            Combine_ = Averaging;
        } else if (!this->ParameterList_->get("Overlapping Operator Combination","Restricted").compare("Full")) {
            Combine_ = Full;
        } else if (!this->ParameterList_->get("Overlapping Operator Combination","Restricted").compare("Restricted")) {
            Combine_ = Restricted;
        }
//        if ( this->MpiComm_->getRank() < this->MpiComm_->getSize() - this->ParameterList_->get("Mpi Ranks Coarse",0)) {
//            OnFirstLevelComm_ = true;
//        }
        if ( this->MpiComm_->getRank() >= this->RankRange_[0] && this->MpiComm_->getRank() <= this->RankRange_[1] )
            OnFirstLevelComm_ = true;

        
//        std::cout << this->MpiComm_->getRank() << " OnFirstLevelComm_:" << OnFirstLevelComm_ << " this->RankRange_[0]:"<<this->RankRange_[0] << " this->RankRange_[1]:" << this->RankRange_[1] << std::endl;
        FirstLevelSolveComm_ = this->MpiComm_->split(!OnFirstLevelComm_,this->MpiComm_->getRank());
    }
    
    template <class SC,class LO,class GO,class NO>
    OverlappingOperator<SC,LO,GO,NO>::~OverlappingOperator()
    {
        SubdomainSolver_.reset();
    }
    
    // Y = alpha * A^mode * X + beta * Y
    template <class SC,class LO,class GO,class NO>
    void OverlappingOperator<SC,LO,GO,NO>::apply(const MultiVector &x,
                                                 MultiVector &y,
                                                 bool usePreconditionerOnly,
                                                 Teuchos::ETransp mode,
                                                 SC alpha,
                                                 SC beta) const
    {
        
        Teuchos::RCP<Teuchos::FancyOStream> fancy = fancyOStream(Teuchos::rcpFromRef(std::cout));
        
        FROSCH_ASSERT(this->IsComputed_,"ERROR: OverlappingOperator has to be computed before calling apply()");

#ifdef FROSCH_TIMER
#ifdef FROSCH_DETAIL_TIMER
        this->MpiComm_->barrier();
#endif
        TimeMonitor_Type ApplyTM(*ApplyTimer_);
#endif
        MultiVectorPtr xTmp = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());
        *xTmp = x;
        
        if (!usePreconditionerOnly && mode == Teuchos::NO_TRANS) {
            this->K_->apply(x,*xTmp,mode,1.0,0.0);
        }
        
        MultiVectorPtr xOverlap;

        MultiVectorPtr yOverlap = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,x.getNumVectors());

        extend(xTmp,xOverlap);
        
//        std::cout <<" xOverlapping " << std::endl;
//        xOverlap->describe(*fancy,Teuchos::VERB_EXTREME);

        if (OnFirstLevelComm_) {
            yOverlap->replaceMap(OverlappingMatrix_->getRangeMap());
            SubdomainSolver_->apply(*xOverlap,*yOverlap,mode,1.0,0.0);
        }

        yOverlap->replaceMap(OverlappingMap_);
        xTmp->putScalar(0.0);
        
        combine(yOverlap,xTmp);
        
//        std::cout <<" yOverlapping " << std::endl;
//        xTmp->describe(*fancy,Teuchos::VERB_EXTREME);
        
        if (!usePreconditionerOnly && mode != Teuchos::NO_TRANS) {
            this->K_->apply(*xTmp,*xTmp,mode,1.0,0.0);
        }
        y.update(alpha,*xTmp,beta);
#ifdef FROSCH_DETAIL_TIMER
        this->MpiComm_->barrier();
#endif
    }
    
    template <class SC,class LO,class GO,class NO>
    void OverlappingOperator<SC,LO,GO,NO>::extend(MultiVectorPtr x, MultiVectorPtr &xOverlap) const
    {
        // AH 11/28/2018: replaceMap does not update the GlobalNumRows. Therefore, we have to create a new MultiVector on the serial Communicator. In Epetra, we can prevent to copy the MultiVector.
        if (x->getMap()->lib() == Xpetra::UseEpetra) {
            XOverlapEpetra_ = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,x->getNumVectors());
            
            XOverlapEpetra_->doImport(*x,*Scatter_,Xpetra::INSERT);
            
            const Teuchos::RCP<const Xpetra::EpetraMultiVectorT<GO,NO> > xEpetraMultiVectorXOverlapTmp = Teuchos::rcp_dynamic_cast<const Xpetra::EpetraMultiVectorT<GO,NO> >(XOverlapEpetra_);
            Teuchos::RCP<Epetra_MultiVector> epetraMultiVectorXOverlapTmp = xEpetraMultiVectorXOverlapTmp->getEpetra_MultiVector();
            
            if (OnFirstLevelComm_) {
                const Teuchos::RCP<const Xpetra::EpetraMapT<GO,NO> >& xEpetraMap = Teuchos::rcp_dynamic_cast<const Xpetra::EpetraMapT<GO,NO> >(OverlappingMatrix_->getRangeMap());
                Epetra_BlockMap epetraMap = xEpetraMap->getEpetra_BlockMap();
                
                double *A;
                int MyLDA;
                epetraMultiVectorXOverlapTmp->ExtractView(&A,&MyLDA);
                
                Teuchos::RCP<Epetra_MultiVector> epetraMultiVectorXOverlap(new Epetra_MultiVector(View,epetraMap,A,MyLDA,x->getNumVectors()));
                xOverlap = Teuchos::RCP<Xpetra::EpetraMultiVectorT<GO,NO> >(new Xpetra::EpetraMultiVectorT<GO,NO>(epetraMultiVectorXOverlap));
            }
        } else {
            xOverlap = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,x->getNumVectors());
            
            xOverlap->doImport(*x,*Scatter_,Xpetra::INSERT);
            if (OnFirstLevelComm_) {
                xOverlap->replaceMap(OverlappingMatrix_->getDomainMap());
            }
        }
    }
    
    template <class SC,class LO,class GO,class NO>
    void OverlappingOperator<SC,LO,GO,NO>::combine(MultiVectorPtr yOverlap, MultiVectorPtr y) const
    {
        if (Combine_ == Restricted){
#ifdef FROSCH_DETAIL_TIMER
            this->MpiComm_->barrier();
            TimeMonitor_Type ApplyRestTM(*ApplyRestTimer_);
#endif
            y->doImport(*yOverlap,*GatherRestricted_,Xpetra::INSERT);
            
#ifdef FROSCH_DETAIL_TIMER
            this->MpiComm_->barrier();
#endif
        }
        else{
#ifdef FROSCH_DETAIL_TIMER
            this->MpiComm_->barrier();
            TimeMonitor_Type ApplyNormalTM(*ApplyNormalTimer_);
#endif
            y->doExport(*yOverlap,*Scatter_,Xpetra::ADD);
#ifdef FROSCH_DETAIL_TIMER
            this->MpiComm_->barrier();
#endif
        }
        if (Combine_ == Averaging) {
            ConstSCVecPtr scaling = Multiplicity_->getData(0);
            for (unsigned j=0; j<y->getNumVectors(); j++) {
                SCVecPtr values = y->getDataNonConst(j);
                for (unsigned i=0; i<values.size(); i++) {
                    values[i] = values[i] / scaling[i];
                }
            }
        }
    }
    
    template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::initializeOverlappingOperator()
    {
#ifdef FROSCH_TIMER
        TimeMonitor_Type OverlapTM(*OverlapTimer_);
        TimeMonitor_Type FullTM(*FullSetupTimer_);
#endif
        // subcom for parallel SumOperator
        Scatter_ = Xpetra::ImportFactory<LO,GO,NO>::Build(this->getDomainMap(),OverlappingMap_);
        if (Combine_ == Averaging) {
            Multiplicity_ = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(this->getRangeMap(),1);
            MultiVectorPtr multiplicityRepeated;
            multiplicityRepeated = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,1);
            multiplicityRepeated->putScalar(1.);
            ExporterPtr multiplicityExporter = Xpetra::ExportFactory<LO,GO,NO>::Build(multiplicityRepeated->getMap(),this->getRangeMap());
            Multiplicity_->doExport(*multiplicityRepeated,*multiplicityExporter,Xpetra::ADD);
        }
        else if(Combine_ == Restricted)
            GatherRestricted_ = Xpetra::ExportFactory<LO,GO,NO>::Build(this->getDomainMap(),OverlappingMap_);
        
//                }
        return 0; // RETURN VALUE
    }
    
    template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::computeOverlappingOperator()
    {
        int ret = 0;
        if (this->IsComputed_) {// already computed once and we want to recycle the information. That is why we reset OverlappingMatrix_ to K_, because K_ has been reset at this point
            OverlappingMatrix_ = this->K_;
        }
#ifdef FROSCH_TIMER
        TimeMonitor_Type FullTM(*FullSetupTimer_);
#endif
        Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
        {
#ifdef FROSCH_TIMER
            TimeMonitor_Type OverlapTM(*ExtractTimer_);
#endif
//            Teuchos::RCP<Epetra_MpiComm> epetraMpiComm(new Epetra_MpiComm(MPI_COMM_WORLD));
//            Teuchos::RCP<Epetra_CrsMatrix> epertaMat = ConvertToEpetra(*(OverlappingMatrix_), epetraMpiComm);
//            EpetraExt::RowMatrixToMatlabFile("K.dat",*epertaMat);
            
            OverlappingMatrix_ = ExtractLocalSubdomainMatrix(OverlappingMatrix_,OverlappingMap_,OnFirstLevelComm_);

//            Teuchos::RCP<Teuchos::FancyOStream> fancy = fancyOStream(Teuchos::rcpFromRef(std::cout));
//
//            OverlappingMap_->describe(*fancy,Teuchos::VERB_EXTREME);
//            
//            Teuchos::RCP<Epetra_MpiComm> epetraSerialComm(new Epetra_MpiComm(MPI_COMM_SELF));
//            Teuchos::RCP<Epetra_CrsMatrix> epertaMatSub = ConvertToEpetra(*(OverlappingMatrix_), epetraSerialComm);
//            std::string outName = "K" + std::to_string( this->MpiComm_->getRank() ) + ".dat";
//            EpetraExt::RowMatrixToMatlabFile(outName.c_str(),*epertaMatSub);

        }
        
#ifdef FROSCH_TIMER
        TimeMonitor_Type OverlapTM(*ComputeTimer_);
#endif
        if (this->IsComputed_) {
            if (this->Verbose_)
                std::cout << "\t### Overlapping Operator(" << LevelID_ << ") does use overlapping maps of previous compute." << std::endl;
            if (this->ParameterList_->sublist("Solver").get("Reuse Symbolic Factorization",false)==false) {
                if (this->Verbose_)
                    std::cout << "\t### Overlapping Operator(" << LevelID_ << ") is not reusing symbolic factorization." << std::endl;
                if (OnFirstLevelComm_) {
                    SubdomainSolver_.reset(new SubdomainSolver<SC,LO,GO,NO>(OverlappingMatrix_,sublist(this->ParameterList_,"Solver")));
                    {
#ifdef FROSCH_TIMER
                        TimeMonitor_Type SymbolicFacTM(*SymbolicFacTimer_);
#endif
                        SubdomainSolver_->initialize();
                    }
                    {
#ifdef FROSCH_TIMER
                        TimeMonitor_Type NumericFacTM(*NumericFacTimer_);
#endif
                        ret = SubdomainSolver_->compute();
                    }
                }
            }
            else{
                if (this->Verbose_)
                    std::cout << "\t### Overlapping Operator(" << LevelID_ << ") is reusing prior symbolic factorization." << std::endl;
                if (OnFirstLevelComm_) {
                    SubdomainSolver_->resetMatrix(OverlappingMatrix_);
                    {
#ifdef FROSCH_TIMER
                    TimeMonitor_Type NumericFacTM(*NumericFacTimer_);
#endif
                        ret = SubdomainSolver_->compute();
                    }
                }
            }
        }
        else {
            if (this->Verbose_)
                std::cout << "\t### Overlapping Operator(" << LevelID_ << ") does not use overlapping maps of previous compute." << std::endl;

             if (OnFirstLevelComm_) {
                SubdomainSolver_.reset(new SubdomainSolver<SC,LO,GO,NO>(OverlappingMatrix_,sublist(this->ParameterList_,"Solver")));
                 {
#ifdef FROSCH_TIMER
                     TimeMonitor_Type SymbolicFacTM(*SymbolicFacTimer_);
#endif
                     SubdomainSolver_->initialize();
                 }
                 {
#ifdef FROSCH_TIMER
                     TimeMonitor_Type NumericFacTM(*NumericFacTimer_);
#endif
                     ret = SubdomainSolver_->compute();
                 }

             }
        }
    
        return ret; // RETURN VALUE
    }
    
    template <class SC,class LO,class GO,class NO>
    typename OverlappingOperator<SC,LO,GO,NO>::MapPtr OverlappingOperator<SC,LO,GO,NO>::getOverlappingMap()
    {
        return OverlappingMap_;
    }


    template <class SC,class LO,class GO,class NO>
    typename OverlappingOperator<SC,LO,GO,NO>::CrsMatrixPtr OverlappingOperator<SC,LO,GO,NO>::getOverlappingMatrix()
    {
        return OverlappingMatrix_;
    }
    
    template <class SC,class LO,class GO,class NO>
    typename OverlappingOperator<SC,LO,GO,NO>::SubdomainSolverPtr OverlappingOperator<SC,LO,GO,NO>::getSubdomainSolver()
    {
        return SubdomainSolver_;
    }
    
    template <class SC,class LO,class GO,class NO>
    typename OverlappingOperator<SC,LO,GO,NO>::MultiVectorPtr OverlappingOperator<SC,LO,GO,NO>::getMultiplicity()
    {
        return Multiplicity_;
    }
    
    template <class SC,class LO,class GO,class NO>
    typename OverlappingOperator<SC,LO,GO,NO>::ImporterPtr OverlappingOperator<SC,LO,GO,NO>::getScatter()
    {
        return Scatter_;
    }
    
    template <class SC,class LO,class GO,class NO>
    typename OverlappingOperator<SC,LO,GO,NO>::CommPtr OverlappingOperator<SC,LO,GO,NO>::getLevelComm()
    {
        return FirstLevelSolveComm_;
    }

    template <class SC,class LO,class GO,class NO>
    bool OverlappingOperator<SC,LO,GO,NO>::getOnLevelComm()
    {
        return OnFirstLevelComm_;
    }

    
    template <class SC,class LO,class GO,class NO>
    CombinationType OverlappingOperator<SC,LO,GO,NO>::getCombineMode()
    {
        return Combine_;
    }
}

#endif
