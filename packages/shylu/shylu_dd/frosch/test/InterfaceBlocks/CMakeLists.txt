TRIBITS_ADD_EXECUTABLE(
interface_blocks
SOURCES main.cpp
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(InterfaceBlocksCopyFiles
    DEST_FILES ParameterList.xml
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
    EXEDEPS interface_blocks
)

## Epetra
IF(ShyLU_DDFROSch_ENABLE_Epetra AND NOT Tpetra_DefaultNode STREQUAL "Kokkos::Compat::KokkosCudaWrapperNode")
TRIBITS_ADD_TEST(
interface_blocks
NAME epetra_interface_blocks
ARGS "--M=2 --DIM=2 --NB=2 --DPN=2 --ORD=0 --PLIST=ParameterList.xml --USEEPETRA"
COMM mpi
NUM_MPI_PROCS 4
)
ENDIF()



## Tpetra
TRIBITS_ADD_TEST(
interface_blocks
NAME tpetra_interface_blocks
ARGS "--M=2 --DIM=2 --NB=2 --DPN=2 --ORD=0 --PLIST=ParameterList.xml --USETPETRA"
COMM mpi
NUM_MPI_PROCS 4
)
