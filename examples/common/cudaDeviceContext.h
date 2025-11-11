//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OSD_EXAMPLES_COMMON_CUDA_DEVICE_CONTEXT_H
#define OSD_EXAMPLES_COMMON_CUDA_DEVICE_CONTEXT_H

struct ID3D11Device;

class CudaDeviceContext {
public:
    CudaDeviceContext();
    ~CudaDeviceContext();

    /// Initialize cuda device from the current GL context
    bool Initialize();

    /// Initialize cuda device from the ID3D11Device
    bool Initialize(ID3D11Device *device);

    /// Returns true if the cuda device has already been initialized
    bool IsInitialized() const {
        return _initialized;
    }

private:
    bool _initialized;
};

#endif  // OSD_EXAMPLES_COMMON_CUDA_DEVICE_CONTEXT_H
