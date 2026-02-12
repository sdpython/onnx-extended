# Fichier pour trouver et configurer nanobind

include(FetchContent)

set(NANOBIND_VERSION "v2.4.0" CACHE STRING "nanobind version")

message(STATUS "Fetching nanobind ${NANOBIND_VERSION}")

FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        ${NANOBIND_VERSION}
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(nanobind)

# Fonction pour cr�er un module nanobind
function(local_nanobind_add_module target_name)
  nanobind_add_module(${target_name} ${ARGN})
endfunction()

# Fonction pour les modules CUDA avec nanobind
function(cuda_nanobind_add_module target_name)
  nanobind_add_module(${target_name} ${ARGN})
endfunction()

message(STATUS "nanobind found and configured")
