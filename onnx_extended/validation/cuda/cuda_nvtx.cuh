#pragma once

#if defined(ENABLE_NVTX)
#include <nvtx3/nvtx3.hpp>
#define NVTX_SCOPE(msg) nvtx3::scoped_range r{msg};
#else
#define NVTX_SCOPE(msg)
#endif
