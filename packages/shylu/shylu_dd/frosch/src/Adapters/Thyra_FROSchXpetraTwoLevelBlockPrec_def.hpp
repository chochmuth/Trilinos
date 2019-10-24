#ifndef THYRA_FROSCHXPETRATWOLEVELBLOCKPREC_DEF
#define THYRA_FROSCHXPETRATWOLEVELBLOCKPREC_DEF

#include <Thyra_FROSchXpetraTwoLevelBlockPrec_decl.hpp>
#include <Thyra_FROSchLinearOp_decl.hpp>

namespace Thyra {
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    
    //Constructor
    template <class SC, class LO, class GO, class NO>
    Thyra_FROSchXpetraTwoLevelBlockPrec<SC,LO,GO,NO>::Thyra_FROSchXpetraTwoLevelBlockPrec()
    {
        paramList_ = rcp(new Teuchos::ParameterList());
    }
    //-----------------------------------------------------------
    template <class SC, class LO, class GO, class NO>
    bool Thyra_FROSchXpetraTwoLevelBlockPrec<SC,LO,GO,NO>::isCompatible(const LinearOpSourceBase<SC>& fwdOpSrc) const
    {
        const RCP<const LinearOpBase<SC> > fwdOp = fwdOpSrc.getOp();
        //so far only Epetra is allowed
        if (Xpetra::ThyraUtils<SC,LO,GO,NO>::isEpetra(fwdOp)) return true;
        
        return false;
    }
    //--------------------------------------------------------------
    template<class SC, class LO, class GO , class NO>
    RCP<PreconditionerBase<SC> >Thyra_FROSchXpetraTwoLevelBlockPrec<SC,LO,GO,NO>::createPrec() const{
        return Teuchos::rcp(new DefaultPreconditioner<SC>);
        
    }
    //-------------------------------------------------------------
    template<class SC, class LO , class GO, class NO>
    void Thyra_FROSchXpetraTwoLevelBlockPrec<SC,LO,GO,NO>::initializePrec( const Teuchos::RCP<const LinearOpSourceBase<SC> >& fwdOpSrc, PreconditionerBase<SC>* prec, const ESupportSolveUse supportSolveUse ) const{
    
        using Teuchos::rcp_dynamic_cast;
        //Some Typedefs
        typedef Xpetra::ThyraUtils<SC,LO,GO,NO> XpThyUtils;
        typedef Xpetra::CrsMatrix<SC,LO,GO,NO> XpCrsMat;
        typedef Xpetra::Matrix<SC,LO,GO,NO> XpMat;
        typedef Thyra::LinearOpBase<SC> ThyLinOpBase;

                
        //PreCheck
        TEUCHOS_ASSERT(Teuchos::nonnull(fwdOpSrc));
        //TEUCHOS_ASSERT(this->isCompatible(*fwdOpSrc));
        TEUCHOS_ASSERT(prec);

//        RCP<ParameterList> paramList(new ParameterList(*paramList_)); // AH: Muessen wir diese Kopie machen? Irgendwie wäre es doch besser, wenn man die nicht kopieren müsste, oder?

        // Retrieve wrapped concrete Xpetra matrix from FwdOp
        const RCP<const ThyLinOpBase> fwdOp = fwdOpSrc->getOp();
        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(fwdOp));

        // Check whether it is Epetra/Tpetra
        bool bIsEpetra  = XpThyUtils::isEpetra(fwdOp);
        bool bIsTpetra  = XpThyUtils::isTpetra(fwdOp);
        bool bIsBlocked = XpThyUtils::isBlockedOperator(fwdOp);
        TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra == true  && bIsTpetra == true));
        TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra == bIsTpetra) && bIsBlocked == false);
        TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra != bIsTpetra) && bIsBlocked == true);

        RCP<XpMat> A = Teuchos::null;
        RCP<const XpCrsMat > xpetraFwdCrsMat = XpThyUtils::toXpetra(fwdOp);
        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMat));
        
        // FROSCH needs a non-const object as input
        RCP<XpCrsMat> xpetraFwdCrsMatNonConst = Teuchos::rcp_const_cast<XpCrsMat>(xpetraFwdCrsMat);
        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMatNonConst));

        // wrap the forward operator as an Xpetra::Matrix that FROSch can work with
        A = rcp(new Xpetra::CrsMatrixWrap<SC,LO,GO,NO>(xpetraFwdCrsMatNonConst));
        
        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(A));
        
        // Retrieve concrete preconditioner object--->Here Mem Leak?
        const Teuchos::Ptr<DefaultPreconditioner<SC> > defaultPrec = Teuchos::ptr(dynamic_cast<DefaultPreconditioner<SC> *>(prec));
        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(defaultPrec));

        // extract preconditioner operator
        RCP<ThyLinOpBase> thyra_precOp = Teuchos::null;
        thyra_precOp = rcp_dynamic_cast<Thyra::LinearOpBase<SC> >(defaultPrec->getNonconstUnspecifiedPrecOp(), true);
        
        //-------Build New Two Level Prec or Recycling of old information --------------
        if ( paramList_->get("Recycle now", false) ) {
            RCP<FROSchLinearOp<SC, LO, GO, NO> > fROSch_LinearOp = Teuchos::null;
            fROSch_LinearOp = rcp_dynamic_cast<FROSchLinearOp<SC, LO, GO, NO> >(thyra_precOp, true);
            
//            fROSch_LinearOp = Teuchos::ptr(dynamic_cast<FROSchLinearOp<SC, LO, GO, NO> *>(defaultPrec));
            
            RCP<  Xpetra::Operator< > > xpetraOp = fROSch_LinearOp->getXpetraOperator();//getConstXpetraOperator();
            
            RCP< FROSch::TwoLevelBlockPreconditioner<SC,LO,GO,NO> > twoLevelBlock = Teuchos::null;
            
            twoLevelBlock = rcp_dynamic_cast< FROSch::TwoLevelBlockPreconditioner<SC,LO,GO,NO> >(xpetraOp, true);
            
            twoLevelBlock->resetMatrix(A);

            twoLevelBlock->compute();

        }
        else {
            RCP<FROSch::TwoLevelBlockPreconditioner<SC,LO,GO,NO> > TwoLevelPrec (new FROSch::TwoLevelBlockPreconditioner<SC,LO,GO,NO>(A,paramList_));
            
            RCP< const Teuchos::Comm< int > > comm = A->getRowMap()->getComm();
            
            //        Teuchos::RCP<Xpetra::MultiVector<SC,LO,GO,NO> > coord = Teuchos::null;
            //
            //        if(paramList->isParameter("Coordinates")){
            //            coord = FROSch::ExtractCoordinatesFromParameterList<SC,LO,GO,NO>(*paramList);
            //        }
            
            
            UN nmbBlocks = paramList_->get("Number of blocks",1);
            MapPtrVecPtr repeatedMapVec(nmbBlocks);
            UNVecPtr dofsPerNodeVec(nmbBlocks);
            GOVecPtr blockMaxGID(nmbBlocks);
            DofOrderingVecPtr dofOrderingVec(nmbBlocks);
            
            GO offsetAllPrior = 0;
            for (UN i=0; i<nmbBlocks; i++) {
                
                std::string repeatedMapName = "RepeatedMap" + std::to_string(i+1);
                if(paramList_->isParameter(repeatedMapName)){
                    Teuchos::RCP<Xpetra::Map<LO,GO,NO> > repeatedMap = FROSch::ExtractRepeatedMapFromParameterList<LO,GO,NO>(*paramList_, repeatedMapName );
                    
                    Teuchos::ArrayView< const GO > 	nodeList = repeatedMap->getNodeElementList();
                    Teuchos::Array<GO> nodeListOffset(nodeList.size());
                    
                    for (unsigned j=0; j<nodeList.size(); j++) {
                        nodeListOffset[j] = nodeList[j] + offsetAllPrior;
                    }
                    repeatedMapVec[i] = Xpetra::MapFactory<LO,GO,NO>::Build(repeatedMap->lib(),-1,nodeListOffset,0,comm);
                    offsetAllPrior = repeatedMapVec[i]->getMaxAllGlobalIndex()+1;
                }
                else{
#ifdef FROSCH_ASSERT
                    FROSCH_ASSERT(false, repeatedMapName + " not found!");
#endif
                }
                
#ifdef FROSCH_ASSERT
                FROSCH_ASSERT(repeatedMapVec[i]!=Teuchos::null,repeatedMapName + " not loaded correctly!");
#endif
                Teuchos::RCP<Teuchos::FancyOStream> fancy = fancyOStream(Teuchos::rcpFromRef(std::cout));
                blockMaxGID[i] = repeatedMapVec[i]->getMaxAllGlobalIndex();
                dofsPerNodeVec[i] = paramList_->get("DofsPerNode" + std::to_string(i+1),1);
                
                std::string ordering = paramList_->get("DofOrdering" + std::to_string(i+1), "NodeWise");
                if (!ordering.compare("NodeWise"))
                    dofOrderingVec[i] = FROSch::NodeWise;
                else if (!ordering.compare("DimensionWise"))
                    dofOrderingVec[i] = FROSch::DimensionWise;
                else if (!ordering.compare("Custom"))
                    dofOrderingVec[i] = FROSch::Custom;
                else
                    FROSCH_ASSERT(false,"ERROR: Specify a valid DofOrdering.");
                
            }
            
            if (comm->getRank()==0) std::cout << "INITIALIZE FROSch...";
            TwoLevelPrec->initialize(paramList_->get("Dimension",2), dofsPerNodeVec, dofOrderingVec,blockMaxGID, paramList_->get("Overlap",1), repeatedMapVec);
            
            TwoLevelPrec->compute();
            //-----------------------------------------------
            
            RCP<ThyLinOpBase > thyraPrecOp = Teuchos::null;
            //FROSCh_XpetraOP
            //        RCP<FROSch_XpetraOperator<SC,LO,GO,NO> > froschXOP (new FROSch_XpetraOperator<SC,LO,GO,NO>(TwoLevelPrec));
            
            RCP<const VectorSpaceBase<SC> > thyraRangeSpace  = Xpetra::ThyraUtils<SC,LO,GO,NO>::toThyra(TwoLevelPrec->getRangeMap());
            RCP<const VectorSpaceBase<SC> > thyraDomainSpace = Xpetra::ThyraUtils<SC,LO,GO,NO>::toThyra(TwoLevelPrec->getDomainMap());
            
            thyraPrecOp = Thyra::fROSchLinearOp<SC, LO, GO, NO>(thyraRangeSpace, thyraDomainSpace,TwoLevelPrec,bIsEpetra,bIsTpetra);
            
            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(thyraPrecOp));
            
            defaultPrec->initializeUnspecified(thyraPrecOp);
        }
        
    }
    
    //-------------------------------------------------------------
    template <class SC, class LO, class GO, class NO>
    void Thyra_FROSchXpetraTwoLevelBlockPrec<SC,LO,GO,NO>::
    uninitializePrec(PreconditionerBase<SC>* prec, RCP<const LinearOpSourceBase<SC> >* fwdOp, ESupportSolveUse* supportSolveUse) const {
        TEUCHOS_ASSERT(prec);
        
        // Retrieve concrete preconditioner object
        const Teuchos::Ptr<DefaultPreconditioner<SC> > defaultPrec = Teuchos::ptr(dynamic_cast<DefaultPreconditioner<SC> *>(prec));
        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(defaultPrec));
        
        if (fwdOp) {
            // TODO: Implement properly instead of returning default value
            *fwdOp = Teuchos::null;
        }
        
        if (supportSolveUse) {
            // TODO: Implement properly instead of returning default value
            *supportSolveUse = Thyra::SUPPORT_SOLVE_UNSPECIFIED;
        }
        
        defaultPrec->uninitialize();
    }
    //-----------------------------------------------------------------
    template <class SC, class LO, class GO, class NO>
    void Thyra_FROSchXpetraTwoLevelBlockPrec<SC,LO,GO,NO>::setParameterList(RCP<ParameterList> const & paramList){
        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(paramList));
        paramList_ = paramList;
    }
    
    //------------------------------------------------------------------
    template <class SC, class LO, class GO, class NO>
    RCP<ParameterList> Thyra_FROSchXpetraTwoLevelBlockPrec<SC,LO,GO,NO>::getNonconstParameterList(){
        return paramList_;
        
    }
    
    //-----------------------------------------------------------------------
    template <class SC, class LO, class GO, class NO>
    RCP<const ParameterList>
    Thyra_FROSchXpetraTwoLevelBlockPrec<SC,LO,GO,NO>::getParameterList() const {
        return paramList_;
    }
    //--------------------------------------------------------------------
    template <class SC, class LO, class GO, class NO>
    RCP<const ParameterList> Thyra_FROSchXpetraTwoLevelBlockPrec<SC,LO,GO,NO>::getValidParameters() const {
        static RCP<const ParameterList> validPL;
        
        if (Teuchos::is_null(validPL))
        validPL = rcp(new ParameterList());
        
        return validPL;
    }
    //-----------------------------------------------------------------------
    template <class SC, class LO, class GO, class NO>
    std::string Thyra_FROSchXpetraTwoLevelBlockPrec<SC,LO,GO,NO>::description() const {
        return "Thyra::Thyra_FROSchXpetraTwoLevelBlockPrec";
    }
    //--------------------------------------------------------------------------
    template<class SC, class LO,class GO, class NO>
    RCP<ParameterList> Thyra_FROSchXpetraTwoLevelBlockPrec<SC, LO,GO,NO>::unsetParameterList(){
        RCP<ParameterList> savedParamList = paramList_;
        paramList_ = Teuchos::null;
        return savedParamList;
        
    }
    
}
#endif


