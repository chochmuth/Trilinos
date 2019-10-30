TRIBITS_ADD_EXECUTABLE(
thyra_xpetra_laplace_par
SOURCES main.cpp
)

ADD_SUBDIRECTORIES(
ParameterLists
)

MESSAGE(STATUS "FROSch tests running with Tpetra_DefaultNode type ${Tpetra_DefaultNode}")

#######
# TLP #
#######
## Epetra
IF(HAVE_SHYLU_DDFROSCH_EPETRA AND NOT Tpetra_DefaultNode STREQUAL "Kokkos::Compat::KokkosCudaWrapperNode")
TRIBITS_ADD_TEST(
thyra_xpetra_laplace_par
NAME test_thyra_xpetra_laplace_TLP_GDSW_DIM2_DPN1_ORD0_EPETRA_PAR
ARGS "--M=3 --DIM=2 --O=1 --NB=1 --DPN=1 --ORD=0 --PLIST=ParameterLists/ParameterList_TwoLevelBlockPreconditioner_GDSW.xml --USEEPETRA"
COMM mpi
NUM_MPI_PROCS 5
)
ENDIF()

## Tpetra
TRIBITS_ADD_TEST(
thyra_xpetra_laplace_par
NAME test_thyra_xpetra_laplace_TLP_GDSW_DIM2_DPN1_ORD0_TPETRA_PAR
ARGS "--M=3 --DIM=2 --O=1 --NB=1 --DPN=1 --ORD=0 --PLIST=ParameterLists/ParameterList_TwoLevelBlockPreconditioner_GDSW.xml --USETPETRA"
COMM mpi
NUM_MPI_PROCS 5
)
