#
# initialization
#
# output variables Eigen_FOUND, EIGEN_TARGET

find_package (Eigen3 REQUIRED NO_MODULE)
set(EIGEN_TARGET Eigen3::Eigen)
set(Eigen_VERSION ${Eigen3_VERSION})


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Eigen
  VERSION_VAR Eigen_VERSION
  REQUIRED_VARS EIGEN_TARGET)
