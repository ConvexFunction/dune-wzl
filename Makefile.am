# $Id$

# we need the module file to be able to build via dunecontrol
EXTRA_DIST = dune.module \
  CMakeLists.txt \
  config.h.cmake \
  dune-wzl-config.cmake.in \
  dune-wzl-version.cmake.in

SUBDIRS = src m4 dune doc cmake

if BUILD_DOCS
# TODO: set up documentation tree automatically
#SUBDIRS += doc
endif

# don't follow the full GNU-standard
# we need automake 1.9 or newer
AUTOMAKE_OPTIONS = foreign 1.9

# pass most important options when "make distcheck" is used
DISTCHECK_CONFIGURE_FLAGS = --with-dune-common=$(DUNE_COMMON_ROOT) --with-dune-geometry=$(DUNE_GEOMETRY_ROOT) --with-dune-grid=$(DUNE_GRID_ROOT) --with-dune-istl=$(DUNE_ISTL_ROOT) --with-dune-localfunctions=$(DUNE_LOCALFUNCTIONS_ROOT) --with-dune-typetree=$(DUNE_TYPETREE_ROOT)  CXX="$(CXX)" CC="$(CC)"

include $(top_srcdir)/am/top-rules
include $(top_srcdir)/am/global-rules

