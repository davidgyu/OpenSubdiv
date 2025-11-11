//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_SHADER_CACHE_H
#define OPENSUBDIV_EXAMPLES_SHADER_CACHE_H

#include <map>

template <typename DESC_TYPE, typename CONFIG_TYPE>
class ShaderCacheT {
public:
    typedef DESC_TYPE DescType;
    typedef CONFIG_TYPE ConfigType;
    typedef std::map<DescType, ConfigType *> ConfigMap;

    virtual ~ShaderCacheT() {
        Reset();
    }

    void Reset() {
        for (typename ConfigMap::iterator it = _configMap.begin();
             it != _configMap.end(); ++it) {
            delete it->second;
        }
        _configMap.clear();
    }

    // fetch shader config
    ConfigType * GetDrawConfig(DescType const & desc) {
        typename ConfigMap::iterator it = _configMap.find(desc);
        if (it != _configMap.end()) {
            return it->second;
        } else {
            ConfigType * config = CreateDrawConfig(desc);
            if (config) {
                _configMap[desc] = config;
            }
            return config;
        }
    }

    virtual ConfigType *CreateDrawConfig(DescType const &desc) = 0;

private:
    ConfigMap _configMap;
};


#endif  // OPENSUBDIV_EXAMPLES_SHADER_CACHE_H
