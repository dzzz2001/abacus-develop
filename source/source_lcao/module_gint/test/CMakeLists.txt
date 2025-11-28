remove_definitions(-D__MPI)

AddTest(
  TARGET MODULE_GINT_helper
  LIBS parameter ${math_libs} base device
  SOURCES gint_helper_test.cpp
)

AddTest(
  TARGET MODULE_GINT_biggrid_info
  LIBS parameter ${math_libs} base device
  SOURCES biggrid_info_test.cpp ../biggrid_info.cpp
)

AddTest(
  TARGET MODULE_GINT_unitcell_info
  LIBS parameter ${math_libs} base device
  SOURCES unitcell_info_test.cpp ../unitcell_info.cpp ../biggrid_info.cpp
)

AddTest(
  TARGET MODULE_GINT_localcell_info
  LIBS parameter ${math_libs} base device
  SOURCES localcell_info_test.cpp ../localcell_info.cpp ../unitcell_info.cpp ../biggrid_info.cpp
)
