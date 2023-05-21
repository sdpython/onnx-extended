#pragma once

#if _WIN32
#if not defined(_MSC_VER)
#define _MSC_VER
#include <dlpack/dlpack.h>
#undef _MSC_VER
#endif
#else
#include <dlpack/dlpack.h>
#endif
