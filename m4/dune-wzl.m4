dnl -*- autoconf -*-
# Macros needed to find dune-wzl and dependent libraries.  They are called by
# the macros in ${top_src_dir}/dependencies.m4, which is generated by
# "dunecontrol autogen"

# Additional checks needed to build dune-wzl
# This macro should be invoked by every module which depends on dune-wzl, as
# well as by dune-wzl itself
AC_DEFUN([DUNE_WZL_CHECKS],[
  AC_REQUIRE([DUNE_PATH_HDF5])
])

# Additional checks needed to find dune-wzl
# This macro should be invoked by every module which depends on dune-wzl, but
# not by dune-wzl itself
AC_DEFUN([DUNE_WZL_CHECK_MODULE],
[
  DUNE_CHECK_MODULES([dune-wzl],[wzl/wzl.hh])
])
