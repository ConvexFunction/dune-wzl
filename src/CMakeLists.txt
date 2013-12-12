
add_executable("dune_wzl" dune_wzl.cc)
target_link_dune_default_libraries("dune_wzl")

add_dune_ug_flags(dune_wzl)
add_dune_alugrid_flags(dune_wzl)
add_dune_umfpack_flags(dune_wzl)
