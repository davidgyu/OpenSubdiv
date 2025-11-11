//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OSD_EXAMPLES_COMMON_CL_DEVICE_CONTEXT_H
#define OSD_EXAMPLES_COMMON_CL_DEVICE_CONTEXT_H

#include <opensubdiv/osd/opencl.h>

class CLDeviceContext {
public:
    CLDeviceContext();
    ~CLDeviceContext();

    static bool HAS_CL_VERSION_1_1 ();

    bool Initialize();

    bool IsInitialized() const {
        return (_clContext != NULL);
    }

    cl_context GetContext() const {
        return _clContext;
    }
    cl_command_queue GetCommandQueue() const {
        return _clCommandQueue;
    }

protected:
    cl_context _clContext;
    cl_command_queue _clCommandQueue;
};

struct ID3D11DeviceContext;

class CLD3D11DeviceContext : public CLDeviceContext {
public:
    bool Initialize(ID3D11DeviceContext *deviceContext);

    ID3D11DeviceContext *GetDeviceContext() const {
        return _d3dDeviceContext;
    }

private:
    ID3D11DeviceContext *_d3dDeviceContext;
};


#endif  // OSD_EXAMPLES_COMMON_CL_DEVICE_CONTEXT_H
