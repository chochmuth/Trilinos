TRIBITS_ADD_EXECUTABLE(
buildNodeMaps
SOURCES main.cpp
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(BuildNodeMapsCopyFiles
    DEST_FILES ParameterList.xml
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
    EXEDEPS buildNodeMaps
)

## Epetra
IF(ShyLU_DDFROSch_ENABLE_Epetra AND NOT Tpetra_DefaultNode STREQUAL "Kokkos::Compat::KokkosCudaWrapperNode")
TRIBITS_ADD_TEST(
buildNodeMaps
NAME epetra_buildNodeMaps
ARGS "--NNE=4 --PLIST=ParameterList.xml --USEEPETRA"
COMM mpi
NUM_MPI_PROCS 2
)
ENDIF()



## Tpetra
TRIBITS_ADD_TEST(
buildNodeMaps
NAME tpetra_buildNodeMaps
ARGS "--NNE=4 --PLIST=ParameterList.xml --USETPETRA"
COMM mpi
NUM_MPI_PROCS 2
)
