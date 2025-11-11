//
//   Copyright 2021 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_BFR_HASH_H
#define OPENSUBDIV3_BFR_HASH_H

#include "../version.h"

#include <cstdint>
#include <cstddef>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Bfr {
namespace internal {

//
//  Internal functions to hash data to unsigned ints for caching. Both
//  32- and 64-bit versions are provided here, but only 64-bit versions
//  are currently intended for internal use (so consider removing 32).
//
//  To compute a hash value for data that is not contiguous in memory,
//  iterate over all the contiguous blocks of memory and accumulate the
//  hash value by passing it on as a seed.  Note that this is *not*
//  equivalent to hashing the contiguous pieces as a whole.  Support
//  for that may be added in future.
// 
uint32_t Hash32(const void *data, size_t len);
uint32_t Hash32(const void *data, size_t len, uint32_t seed);

uint64_t Hash64(const void *data, size_t len);
uint64_t Hash64(const void *data, size_t len, uint64_t seed);

} // end namespace internal
} // end namespace Bfr

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_BFR_HASH_H */
