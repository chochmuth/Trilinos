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
// Questions? Contact Christian Hochmuth (c.hochmuth@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_MULTIPLICATIVEOPERATOR_DEF_HPP
#define _FROSCH_MULTIPLICATIVEOPERATOR_DEF_HPP

#include <FROSch_MultiplicativeOperator_decl.hpp>

namespace FROSch {
    
    template <class SC,class LO,class GO,class NO>
    MultiplicativeOperator<SC,LO,GO,NO>::MultiplicativeOperator(CrsMatrixPtr k, ParameterListPtr parameterList) :
    LevelCombinationOperator<SC,LO,GO,NO> (k, parameterList)
#ifdef FROSCH_TIMER
    ,ApplyMultTimer_(TimeMonitor_Type::getNewCounter("FROSch: Multiplicative Operator: Apply"))
#endif
    {
        
    }
    
    template <class SC,class LO,class GO,class NO>
    MultiplicativeOperator<SC,LO,GO,NO>::MultiplicativeOperator(CrsMatrixPtr k, SchwarzOperatorPtrVecPtr operators, ParameterListPtr parameterList) :
    LevelCombinationOperator<SC,LO,GO,NO> (k, operators, parameterList)
#ifdef FROSCH_TIMER
    ,ApplyMultTimer_(TimeMonitor_Type::getNewCounter("FROSch: Multiplicative Operator: Apply"))
#endif
    {
        
    }
        
    template <class SC,class LO,class GO,class NO>
    MultiplicativeOperator<SC,LO,GO,NO>::~MultiplicativeOperator()
    {
        
    }
        
    
    // Y = alpha * A^mode * X + beta * Y
    template <class SC,class LO,class GO,class NO>
    void MultiplicativeOperator<SC,LO,GO,NO>::apply(const MultiVector &x,
                                         MultiVector &y,
                                         bool usePreconditionerOnly,
                                         Teuchos::ETransp mode,
                                         SC alpha,
                                         SC beta) const
    {

        FROSCH_ASSERT(usePreconditionerOnly,"MultiplicativeOperator can only be used as a preconditioner.");
        FROSCH_ASSERT(this->OperatorVector_.size()==2,"Should be a Two-Level Operator.");

#ifdef FROSCH_TIMER
        
        TimeMonitor_Type ApplyMultTimerTM(*ApplyMultTimer_);
#endif
        MultiVectorPtr xTmp = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());
        *xTmp = x; // Need this for the case when x aliases y

        MultiVectorPtr yTmp = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(y.getMap(),y.getNumVectors());
        *yTmp = y; // for the second apply

        this->OperatorVector_[0]->apply(*xTmp,*yTmp,true);
        
        this->K_->apply(*yTmp,*xTmp);
        
        this->OperatorVector_[1]->apply(*xTmp,*xTmp,true);

        yTmp->update(-1.0,*xTmp,1.0);
        y.update(alpha,*yTmp,beta);
        
    }

    template <class SC,class LO,class GO,class NO>
    std::string MultiplicativeOperator<SC,LO,GO,NO>::description() const
    {
        std::string labelString = "Multiplicative operator: ";
        
        for (UN i=0; i<this->OperatorVector_.size(); i++) {
            labelString += this->OperatorVector_[i]->description();
            if (i<this->OperatorVector_.size()-1) {
                labelString += ",";
            }
        }
        return labelString;
    }
    
}

#endif
