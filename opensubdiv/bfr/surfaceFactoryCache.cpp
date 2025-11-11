//
//   Copyright 2021 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../bfr/surfaceFactoryCache.h"
#include "../bfr/patchTree.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Bfr {

//
//  Trivial constructor and destructor:
//
SurfaceFactoryCache::SurfaceFactoryCache() : _map() {
}

SurfaceFactoryCache::~SurfaceFactoryCache() {
    //  Potentially monitor usage on destruction
}


//
//  Internal methods to find and add map entries:
//
SurfaceFactoryCache::DataType
SurfaceFactoryCache::find(KeyType const & key) const {

    MapType::const_iterator itFound = _map.find(key);
    return (itFound != _map.end()) ? itFound->second : DataType(0);
}

SurfaceFactoryCache::DataType
SurfaceFactoryCache::add(KeyType const & key, DataType const & data) {

    MapType::const_iterator itFound = _map.find(key);
    return (itFound != _map.end()) ? itFound->second : (_map[key] = data);
}

//
//  Virtual method defaults -- intended to be overridden for thread-safety:
//
SurfaceFactoryCache::DataType
SurfaceFactoryCache::Find(KeyType const & key) const {

    return find(key);
}

SurfaceFactoryCache::DataType
SurfaceFactoryCache::Add(KeyType const & key, DataType const & data) {

    return add(key, data);
}

} // end namespace Bfr

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv
