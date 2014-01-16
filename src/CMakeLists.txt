add_executable("dune_wzl" dune_wzl.cc)
target_link_dune_default_libraries("dune_wzl")

add_dune_ug_flags(dune_wzl)
add_dune_alugrid_flags(dune_wzl)
add_dune_umfpack_flags(dune_wzl)


find_package(HDF5 REQUIRED COMPONENTS CXX)

target_compile_options(dune_wzl PUBLIC ${HDF5_DEFINITIONS})
target_include_directories(dune_wzl PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(dune_wzl ${HDF5_LIBRARIES})



add_executable("dune_wzl_compare" dune_wzl_compare.cc)
target_link_dune_default_libraries("dune_wzl_compare")

add_dune_ug_flags(dune_wzl_compare)
add_dune_alugrid_flags(dune_wzl_compare)
add_dune_umfpack_flags(dune_wzl_compare)


find_package(HDF5 REQUIRED COMPONENTS CXX)

target_compile_options(dune_wzl_compare PUBLIC ${HDF5_DEFINITIONS})
target_include_directories(dune_wzl_compare PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(dune_wzl_compare ${HDF5_LIBRARIES})
