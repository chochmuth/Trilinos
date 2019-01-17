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

#ifndef THYRA_FROSCH_LINEAR_OP_DEF_HPP
#define THYRA_FROSCH_LINEAR_OP_DEF_HPP

#include "Thyra_FROSchLinearOp_decl.hpp"
#include <FROSch_Tools_def.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_THYRA

namespace Thyra {
    
    using namespace std;
//    using namespace Belos;
    using namespace FROSch;
    using namespace Teuchos;
    using namespace Xpetra;
    
    // Constructors/initializers
    template <class SC, class LO, class GO, class NO>
    FROSchLinearOp<SC,LO,GO,NO>::FROSchLinearOp()
    {}
    
    template <class SC, class LO, class GO, class NO>
    void FROSchLinearOp<SC,LO,GO,NO>::initialize(const RCP<const VectorSpaceBase<SC> > &rangeSpace,
                                                 const RCP<const VectorSpaceBase<SC> > &domainSpace,
                                                 const RCP<Xpetra::Operator<SC,LO,GO,NO> > &xpetraOperator,
                                                 bool bIsEpetra,
                                                 bool bIsTpetra)
    {
        initializeImpl(rangeSpace, domainSpace, xpetraOperator,bIsEpetra,bIsTpetra);
    }
    
    template <class SC, class LO, class GO, class NO>
    void FROSchLinearOp<SC,LO,GO,NO>::constInitialize(const RCP<const VectorSpaceBase<SC> > &rangeSpace,
                                                      const RCP<const VectorSpaceBase<SC> > &domainSpace,
                                                      const RCP<const Xpetra::Operator<SC,LO,GO,NO> > &xpetraOperator,
                                                      bool bIsEpetra,
                                                      bool bIsTpetra)
    {
        initializeImpl(rangeSpace, domainSpace, xpetraOperator,bIsEpetra,bIsTpetra);
    }
    
    template <class SC, class LO, class GO, class NO>
    RCP<Xpetra::Operator<SC,LO,GO,NO> > FROSchLinearOp<SC,LO,GO,NO>::getXpetraOperator()
    {
        return xpetraOperator_.getNonconstObj();
    }
    
    template <class SC, class LO, class GO, class NO>
    RCP<const Xpetra::Operator<SC,LO,GO,NO> > FROSchLinearOp<SC,LO,GO,NO>::getConstXpetraOperator() const
    {
        return xpetraOperator_;
    }
    
    // Public Overridden functions from LinearOpBase
    
    template <class SC, class LO, class GO, class NO>
    RCP<const VectorSpaceBase<SC> > FROSchLinearOp<SC,LO,GO,NO>::range() const
    {
        return rangeSpace_;
    }
    
    template <class SC, class LO, class GO, class NO>
    RCP<const VectorSpaceBase<SC> > FROSchLinearOp<SC,LO,GO,NO>::domain() const
    {
        return domainSpace_;
    }
    
    // Protected Overridden functions from LinearOpBase
    
    template <class SC, class LO, class GO, class NO>
    bool FROSchLinearOp<SC,LO,GO,NO>::opSupportedImpl(EOpTransp M_trans) const
    {
        if (is_null(xpetraOperator_))
        return false;
        
        if (M_trans == NOTRANS)
        return true;
        
        if (M_trans == CONJ) {
            // For non-complex scalars, CONJ is always supported since it is equivalent to NO_TRANS.
            // For complex scalars, Xpetra does not support conjugation without transposition.
            return !ScalarTraits<SC>::isComplex;
        }
        
        return xpetraOperator_->hasTransposeApply();
    }
    
    template <class SC, class LO, class GO, class NO>
    void FROSchLinearOp<SC,LO,GO,NO>::applyImpl(const EOpTransp M_trans,
                                                const MultiVectorBase<SC> &X_in,
                                                const Ptr<MultiVectorBase<SC> > &Y_inout,
                                                const SC alpha,
                                                const SC beta) const
    {
        const EOpTransp real_M_trans = real_trans(M_trans);

        FROSCH_ASSERT(getConstXpetraOperator()!=Teuchos::null,"XpetraLinearOp::applyImpl: internal Xpetra::Operator is null.");
        RCP< const Comm<int> > comm = getConstXpetraOperator()->getRangeMap()->getComm();
        //Transform to Xpetra MultiVector
        RCP<MultiVector<SC,LO,GO,NO> > xY;
        
        ETransp transp;
        switch (M_trans) {
            case NOTRANS:   transp = Teuchos::NO_TRANS;   break;
            case TRANS:     transp = Teuchos::TRANS;      break;
            case CONJTRANS: transp = Teuchos::CONJ_TRANS; break;
            default: FROSCH_ASSERT(false,"Thyra::XpetraLinearOp::apply. Unknown value for M_trans. Only NOTRANS, TRANS and CONJTRANS are supported.");
        }
        //Epetra NodeType
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
        if(this->bIsEpetra_){
            const RCP<const VectorSpaceBase<double> > XY_domain = X_in.domain();
            
            RCP<const Map<LO,GO,NO> > DomainM = this->xpetraOperator_->getDomainMap();
        
            RCP<const Map<LO,GO,NO> >RangeM = this->xpetraOperator_->getRangeMap();
        
            RCP<const EpetraMapT<GO,NO> > eDomainM = rcp_dynamic_cast<const EpetraMapT<GO,NO> >(DomainM);
        
            const Epetra_Map epetraDomain = eDomainM->getEpetra_Map();
        
            RCP<const EpetraMapT<GO,NO> > eRangeM = rcp_dynamic_cast<const EpetraMapT<GO,NO> >(RangeM);
        
            const Epetra_Map epetraRange = eRangeM->getEpetra_Map();
            
            RCP<const Epetra_MultiVector> X;
        
            RCP<Epetra_MultiVector> Y;
       
            THYRA_FUNC_TIME_MONITOR_DIFF("Thyra::EpetraLinearOp::euclideanApply: Convert MultiVectors", MultiVectors);
            // X
            X = get_Epetra_MultiVector(real_M_trans==NOTRANS ? epetraDomain: epetraRange, X_in );
            RCP<Epetra_MultiVector> X_nonconst = rcp_const_cast<Epetra_MultiVector>(X);
            RCP<MultiVector<SC,LO,GO,NO> > xX = FROSch::ConvertToXpetra<SC,LO,GO,NO>(UseEpetra,*X_nonconst,comm);
            // Y
            Y = get_Epetra_MultiVector(real_M_trans==NOTRANS ? epetraRange: epetraDomain, *Y_inout );
            xY = FROSch::ConvertToXpetra<SC,LO,GO,NO>(UseEpetra,*Y,comm);
            xpetraOperator_->apply(*xX, *xY, transp, alpha, beta);

        } //Tpetra NodeType
        else
#endif
        if(bIsTpetra_){
            const RCP<const MultiVector<SC,LO,GO,NO> > xX = ThyraUtils<SC,LO,GO,NO>::toXpetra(rcpFromRef(X_in), comm);
            xY = ThyraUtils<SC,LO,GO,NO>::toXpetra(rcpFromPtr(Y_inout), comm);
            xpetraOperator_->apply(*xX, *xY, transp, alpha, beta);
            
        } else {
            FROSCH_ASSERT(false,"There is a problem with the underlying lib in FROSchLinearOp.");
            //åstd::cout<<"Only Implemented for Epetra and Tpetra\n";
        }
        
        RCP<MultiVectorBase<SC> >thyraX =
        rcp_const_cast<MultiVectorBase<SC> >(ThyraUtils<SC,LO,GO,NO>::toThyraMultiVector(xY));
        
        typedef SpmdVectorSpaceBase<SC> ThySpmdVecSpaceBase;
        RCP<const ThySpmdVecSpaceBase> mpi_vs = rcp_dynamic_cast<const ThySpmdVecSpaceBase>(rcpFromPtr(Y_inout)->range());
        
        TEUCHOS_TEST_FOR_EXCEPTION(mpi_vs == Teuchos::null, std::logic_error, "Failed to cast Thyra::VectorSpaceBase to Thyra::SpmdVectorSpaceBase.");
        const LO localOffset = ( mpi_vs != Teuchos::null ? mpi_vs->localOffset() : 0 );
        const LO localSubDim = ( mpi_vs != Teuchos::null ? mpi_vs->localSubDim() : rcpFromPtr(Y_inout)->range()->dim() );
        
        RCP<DetachedMultiVectorView<SC> > thyData =
        rcp(new DetachedMultiVectorView<SC>(*rcpFromPtr(Y_inout),Range1D(localOffset,localOffset+localSubDim-1)));
        
        for( size_t j = 0; j <xY->getNumVectors(); ++j) {
            Teuchos::ArrayRCP< const SC > xpData = xY->getData(j); // access const data from Xpetra object
            // loop over all local rows
            for( LO i = 0; i < localSubDim; ++i) {
                (*thyData)(i,j) = xpData[i];
            }
        }
    }
    
    // private
    
    template <class SC, class LO, class GO, class NO>
    template<class XpetraOperator_t>
    void FROSchLinearOp<SC,LO,GO,NO>::initializeImpl(const RCP<const VectorSpaceBase<SC> > &rangeSpace,
                                                     const RCP<const VectorSpaceBase<SC> > &domainSpace,
                                                     const RCP<XpetraOperator_t> &xpetraOperator,
                                                     bool bIsEpetra,
                                                     bool bIsTpetra)
    {
#ifdef THYRA_DEBUG
        TEUCHOS_ASSERT(nonnull(rangeSpace));
        TEUCHOS_ASSERT(nonnull(domainSpace));
        TEUCHOS_ASSERT(nonnull(xpetraOperator));
#endif
        rangeSpace_ = rangeSpace;
        domainSpace_ = domainSpace;
        xpetraOperator_ = xpetraOperator;
        bIsEpetra_ = bIsEpetra;
        bIsTpetra_ = bIsTpetra;
    }
    
} // namespace Thyra

#endif

#endif  // THYRA_XPETRA_LINEAR_OP_HPP
