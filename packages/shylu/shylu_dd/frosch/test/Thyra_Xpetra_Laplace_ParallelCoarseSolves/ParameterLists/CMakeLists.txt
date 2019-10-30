TRIBITS_COPY_FILES_TO_BINARY_DIR(ThyraXpetraLaplaceParCopyFiles
    DEST_FILES  ParameterList_TwoLevelBlockPreconditioner_GDSW.xml
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
    EXEDEPS thyra_xpetra_laplace_par
)
