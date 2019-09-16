/*
// @HEADER
// ***********************************************************************
//
//    Thyra: Interfaces and Support for Abstract Numerical Algorithms
//                 Copyright (2004) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact Roscoe A. Bartlett (bartlettra@ornl.gov)
//
// ***********************************************************************
// @HEADER
*/


#include "Thyra_TpetraThyraWrappers.hpp"
#include "Thyra_VectorSpaceTester.hpp"
#include "Thyra_VectorStdOpsTester.hpp"
#include "Thyra_MultiVectorStdOpsTester.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_LinearOpTester.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_TestingTools.hpp"
#include "Thyra_ScaledLinearOpBase.hpp"
#include "Thyra_RowStatLinearOpBase.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_Tuple.hpp"

#include <Thyra_DefaultProductMultiVector_decl.hpp>
#include <Tpetra_DefaultPlatform.hpp>

#include <Thyra_DefaultProductVectorSpace_decl.hpp>

namespace Thyra {


//
// Helper code and declarations
//


using Teuchos::as;
using Teuchos::null;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::ArrayView;
using Teuchos::rcp_dynamic_cast;
using Teuchos::inOutArg;
using Teuchos::Comm;
using Teuchos::tuple;


bool showAllTests = false;
bool dumpAll = false;
bool runLinearOpTester = true;


TEUCHOS_STATIC_SETUP()
{
  Teuchos::UnitTestRepository::getCLP().setOption(
    "show-all-tests", "no-show-all-tests", &showAllTests, "Show all tests or not" );
  Teuchos::UnitTestRepository::getCLP().setOption(
    "dump-all", "no-dump-all", &dumpAll, "Dump all objects being tested" );
  Teuchos::UnitTestRepository::getCLP().setOption(
    "run-linear-op-tester", "no-run-linear-op-tester", &runLinearOpTester, "..." );
}

//
// TpetraLinearOp
//

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( TpetraThyraWrappers, TpetraMultiVectorsToThyraBlockMV,
  Scalar )
{

    typedef Teuchos::ScalarTraits<Scalar> ST;
    typedef KokkosClassic::DefaultNode::DefaultNodeType Node;
//    using Teuchos::Comm;
//    using Teuchos::RCP;
//    using Teuchos::as;
    using namespace Teuchos;
    
    RCP<const Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
    

    typedef RCP<const VectorSpaceBase<Scalar> > VS_PTR;
    typedef RCP<VectorBase<Scalar> > V_PTR;
    typedef RCP<const ProductVectorSpaceBase<Scalar> > PVS_PTR;
    typedef RCP<ProductMultiVectorBase<Scalar> > PMV_PTR;
    typedef Tpetra::Map<int,int,Node> TPETRAMAP;
    typedef RCP<const TPETRAMAP> TPETRAMAP_PTR;
    typedef Tpetra::MultiVector<Scalar,int,int,Node> TPETRAMV;
    typedef RCP<TPETRAMV> TPETRAMV_PTR;
    
    int numEl = 1;//comm->getRank()==0;
    const int INVALID = Teuchos::OrdinalTraits<int>::invalid();
    
    TPETRAMAP_PTR tpetramap = rcp( new TPETRAMAP( INVALID, numEl, 0, comm ) );

    RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(std::cout));

    tpetramap->describe(*fancy, VERB_EXTREME);

    numEl = comm->getRank()==0;
    TPETRAMAP_PTR tpetramap2 = rcp( new TPETRAMAP( INVALID, numEl, 0, comm ) );
//    Array< VS_PTR > vecSpaces(2);
//    vecSpaces[0] = Thyra::createVectorSpace<Scalar>(tpetramap);
//    vecSpaces[1] = Thyra::createVectorSpace<Scalar>(tpetramap);

    VS_PTR vecSpace = Thyra::createVectorSpace<Scalar>(tpetramap);
    VS_PTR vecSpace2 = Thyra::createVectorSpace<Scalar>(tpetramap2);
//    PVS_PTR prodSpace = productVectorSpace( vecSpaces() );
//    prodSpace->describe(*fancy, VERB_EXTREME);
//    PMV_PTR prodMV_A = defaultProductMultiVector( rcp_dynamic_cast<const DefaultProductVectorSpace<Scalar> >(prodSpace), 1 );
//    PMV_PTR prodMV_B = defaultProductMultiVector( rcp_dynamic_cast<const DefaultProductVectorSpace<Scalar> >(prodSpace), 1 );
//    PMV_PTR prodMV_C = defaultProductMultiVector( rcp_dynamic_cast<const DefaultProductVectorSpace<Scalar> >(prodSpace), 1 );
//    prodMV_A->assign(1.);
//    prodMV_B->assign(1.);
//    prodMV_C->assign(0.);

    V_PTR vecA = createMember(vecSpace);
    V_PTR vecB = createMember(vecSpace);
    V_PTR vecC = createMember(vecSpace2);
   
    vecA->assign(1.);
    vecB->assign(1.);
    vecA->describe(*fancy, VERB_EXTREME);
    
    Thyra::apply<Scalar>(*vecA, Thyra::TRANS, *vecB, vecC.ptr(),1.,0.);
    vecC->describe(*fancy, VERB_EXTREME);
    
    vecB->assign(0.);
    Thyra::apply<Scalar>(*vecA, Thyra::NOTRANS, *vecB, vecC.ptr(),1.,1.);
    
    vecC->describe(*fancy, VERB_EXTREME);
    
//    const RCP<const VectorSpaceBase<Scalar> > rangeSpace =
//    Thyra::createVectorSpace<Scalar>(tpetraOp->getRangeMap());
//

//  const RCP<const VectorSpaceBase<Scalar> > rangeSpace =
//    Thyra::createVectorSpace<Scalar>(tpetraOp->getRangeMap());
//  const RCP<const VectorSpaceBase<Scalar> > domainSpace =
//    Thyra::createVectorSpace<Scalar>(tpetraOp->getDomainMap());
//  const RCP<const LinearOpBase<Scalar> > thyraLinearOp =
//    Thyra::tpetraLinearOp(rangeSpace, domainSpace, tpetraOp);
//  TEST_ASSERT(nonnull(thyraLinearOp));
//
//  out << "\nCheck that operator returns the right thing ...\n";
//  const RCP<VectorBase<Scalar> > x = createMember(thyraLinearOp->domain());
//  Thyra::V_S(x.ptr(), ST::one());
//  const RCP<VectorBase<Scalar> > y = createMember(thyraLinearOp->range());
//  Thyra::apply<Scalar>(*thyraLinearOp, Thyra::NOTRANS, *x, y.ptr());
//  const Scalar sum_y = sum(*y);
//  TEST_FLOATING_EQUALITY( sum_y, as<Scalar>(3+1+2*(y->space()->dim()-2)),
//    100.0 * ST::eps() );

}


//
// createLinearOp
//

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( TpetraThyraWrappers, createLinearOp,
  Scalar )
{

//  typedef Thyra::TpetraOperatorVectorExtraction<Scalar> ConverterT;
//
//  const RCP<Tpetra::Operator<Scalar> > tpetraOp =
//    createTriDiagonalTpetraOperator<Scalar>(g_localDim);
//  out << "tpetraOp = " << Teuchos::describe(*tpetraOp, Teuchos::VERB_HIGH) << std::endl;
//
//  const RCP<const VectorSpaceBase<Scalar> > rangeSpace =
//    Thyra::createVectorSpace<Scalar>(tpetraOp->getRangeMap());
//
//  const RCP<const VectorSpaceBase<Scalar> > domainSpace =
//    Thyra::createVectorSpace<Scalar>(tpetraOp->getDomainMap());
//
//  {
//    const RCP<LinearOpBase<Scalar> > thyraOp =
//      createLinearOp(tpetraOp, rangeSpace, domainSpace);
//    TEST_EQUALITY(thyraOp->range(), rangeSpace);
//    TEST_EQUALITY(thyraOp->domain(), domainSpace);
//    const RCP<Tpetra::Operator<Scalar> > tpetraOp2 =
//      ConverterT::getTpetraOperator(thyraOp);
//    TEST_EQUALITY(tpetraOp2, tpetraOp);
//  }
//
//  {
//    const RCP<LinearOpBase<Scalar> > thyraOp =
//      Thyra::createLinearOp(tpetraOp);
//    TEST_INEQUALITY(thyraOp->range(), rangeSpace);
//    TEST_INEQUALITY(thyraOp->domain(), domainSpace);
//    TEST_ASSERT(thyraOp->range()->isCompatible(*rangeSpace));
//    TEST_ASSERT(thyraOp->domain()->isCompatible(*domainSpace));
//    const RCP<Tpetra::Operator<Scalar> > tpetraOp2 =
//      ConverterT::getTpetraOperator(thyraOp);
//    TEST_EQUALITY(tpetraOp2, tpetraOp);
//  }

}


//
// createConstLinearOp
//

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( TpetraThyraWrappers, createConstLinearOp,
  Scalar )
{

//  typedef Thyra::TpetraOperatorVectorExtraction<Scalar> ConverterT;
//
//  const RCP<const Tpetra::Operator<Scalar> > tpetraOp =
//    createTriDiagonalTpetraOperator<Scalar>(g_localDim);
//  out << "tpetraOp = " << Teuchos::describe(*tpetraOp, Teuchos::VERB_HIGH) << std::endl;
//
//  const RCP<const VectorSpaceBase<Scalar> > rangeSpace =
//    Thyra::createVectorSpace<Scalar>(tpetraOp->getRangeMap());
//
//  const RCP<const VectorSpaceBase<Scalar> > domainSpace =
//    Thyra::createVectorSpace<Scalar>(tpetraOp->getDomainMap());
//
//  {
//    const RCP<const LinearOpBase<Scalar> > thyraOp =
//      createConstLinearOp(tpetraOp, rangeSpace, domainSpace);
//    TEST_EQUALITY(thyraOp->range(), rangeSpace);
//    TEST_EQUALITY(thyraOp->domain(), domainSpace);
//    const RCP<const Tpetra::Operator<Scalar> > tpetraOp2 =
//      ConverterT::getConstTpetraOperator(thyraOp);
//    TEST_EQUALITY(tpetraOp2, tpetraOp);
//  }
//
//  {
//    const RCP<const LinearOpBase<Scalar> > thyraOp =
//      Thyra::createConstLinearOp(tpetraOp);
//    TEST_INEQUALITY(thyraOp->range(), rangeSpace);
//    TEST_INEQUALITY(thyraOp->domain(), domainSpace);
//    TEST_ASSERT(thyraOp->range()->isCompatible(*rangeSpace));
//    TEST_ASSERT(thyraOp->domain()->isCompatible(*domainSpace));
//    const RCP<const Tpetra::Operator<Scalar> > tpetraOp2 =
//      ConverterT::getConstTpetraOperator(thyraOp);
//    TEST_EQUALITY(tpetraOp2, tpetraOp);
//  }

}


//
// Tpetra-implemented methods
//


Teuchos::RCP<Teuchos::Time> lookupAndAssertTimer(const std::string &label)
{
  Teuchos::RCP<Teuchos::Time> timer = Teuchos::TimeMonitor::lookupCounter(label);
  TEUCHOS_TEST_FOR_EXCEPTION(timer == null,
    std::runtime_error,
    "lookupAndAssertTimer(): timer \"" << label << "\" was not present in Teuchos::TimeMonitor."
    " Unit test not valid.");
  return timer;
}


#define CHECK_TPETRA_FUNC_CALL_INCREMENT( timerStr, tpetraCode, thyraCode ) \
{ \
  out << "\nTesting that Thyra calls down to " << timerStr << "\n"; \
  ECHO(tpetraCode); \
  const RCP<const Time> timer = lookupAndAssertTimer(timerStr); \
  const int countBefore = timer->numCalls();  \
  ECHO(thyraCode); \
  const int countAfter = timer->numCalls(); \
  TEST_EQUALITY( countAfter, countBefore+1 ); \
}



#ifdef TPETRA_TEUCHOS_TIME_MONITOR
#  define TPETRA_TIMER_TESTS(SCALAR)  \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( TpetraThyraWrappers, UseTpetraImplementations, SCALAR )
#else
#  define TPETRA_TIMER_TESTS(SCALAR)
#endif




//
// Unit test instantiations
//

#define THYRA_TPETRA_THYRA_WRAPPERS_INSTANT(SCALAR) \
 \
  TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( TpetraThyraWrappers, \
    TpetraMultiVectorsToThyraBlockMV, SCALAR ) \



// We can currently only explicitly instantiate with double support because
// Tpetra only supports explicit instantaition with double.  As for implicit
// instantation, g++ 3.4.6 on my Linux machine was taking more than 30 minutes
// to compile this file when all of the types double, float, complex<double>,
// and complex<float> where enabled.  Therefore, we will only test double for
// now until explicit instantation with other types are supported by Tpetra.

THYRA_TPETRA_THYRA_WRAPPERS_INSTANT(double)


} // namespace Thyra
