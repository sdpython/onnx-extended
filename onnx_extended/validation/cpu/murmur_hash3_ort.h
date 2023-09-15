#pragma once

#if defined(_MSC_VER)

typedef unsigned char uint8_t;
typedef unsigned long uint32_t;
typedef unsigned __int64 uint64_t;

// Other compilers

#else // defined(_MSC_VER)

#include <stdint.h>

#endif // !defined(_MSC_VER)

namespace validation {
namespace ort {

void MurmurHash3_x86_32(const void *key, int len, uint32_t seed, bool is_positive,
                        void *out);

}
} // namespace validation
