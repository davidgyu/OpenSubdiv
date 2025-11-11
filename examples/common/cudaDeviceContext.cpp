//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "cudaDeviceContext.h"

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#elif defined(__APPLE__)
    #include <OpenGL/OpenGL.h>
#else
    #include <X11/Xlib.h>
    #include <GL/glx.h>
#endif

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#if defined(OPENSUBDIV_HAS_DX11SDK)
#include <cuda_d3d11_interop.h>
#endif

#define message(fmt, ...)
//#define message(fmt, ...)  fprintf(stderr, fmt, __VA_ARGS__)
#define error(fmt, ...)  fprintf(stderr, fmt, __VA_ARGS__)

// -----------------------------------------------------------------------
#if CUDA_VERSION < 5000
static int _GetCudaDeviceForCurrentGLContext()
{
#if defined(_WIN32)

    return 0;

#elif defined(__APPLE__)

    return 0;

#else  // X11
    // If we don't have a current GL context, then choose the device which
    // matches the current X11 screen number.
    Display * display = glXGetCurrentDisplay();
    if (!display) {
        display = XOpenDisplay(NULL);
        if (display) {
            int screen = DefaultScreen(display);
            XCloseDisplay(display);
            message("CUDA init using device for default screen: %d\n", screen);
            return screen;
        }
        return 0;
    }

    // We can't use the new interop API, so use the device
    // corresponding to the screen number of the current GL context.
    int screen = DefaultScreen(display);
    message("CUDA init using device for screen: %d\n", screen);
    return screen;
#endif  // X11
}

#else   // CUDA_VERSION >= 50000 -----------------------------------------
static int _GetCudaDeviceForCurrentGLContext()
{
    // Find and use the CUDA device for the current GL context
    unsigned int interopDeviceCount = 0;
    int interopDevices[1];
    cudaError_t status = cudaGLGetDevices(&interopDeviceCount, interopDevices,
                                          1,  cudaGLDeviceListCurrentFrame);
    if (status == cudaErrorNoDevice || interopDeviceCount != 1) {
        message("CUDA no interop devices found.\n");
        return 0;
    }
    int device = interopDevices[0];

#if defined(_WIN32)
    return device;

#elif defined(__APPLE__)
    return device;

#else  // X11
    Display * display = glXGetCurrentDisplay();
    int screen = DefaultScreen(display);
    if (device != screen) {
        error("The CUDA interop device (%d) does not match "
              "the screen used by the current GL context (%d), "
              "which may cause slow performance on systems "
              "with multiple GPU devices.", device, screen);
    }
    message("CUDA init using device for current GL context: %d\n", device);
    return device;
#endif
}
#endif   // CUDA_VERSION -----------------------------------------------

CudaDeviceContext::CudaDeviceContext() :
    _initialized(false) {
}

CudaDeviceContext::~CudaDeviceContext() {
    cudaDeviceReset();
}

bool
CudaDeviceContext::Initialize() {

    // see if any cuda device is available.
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    message("CUDA device count: %d\n", deviceCount);
    if (deviceCount <= 0) {
        return false;
    }

    cudaGLSetGLDevice(_GetCudaDeviceForCurrentGLContext());
    _initialized = true;
    return true;
}

bool
CudaDeviceContext::Initialize(ID3D11Device *device) {

#if defined(OPENSUBDIV_HAS_DX11SDK)
    cudaD3D11SetDirect3DDevice(device);
    return true;
#else
    (void)device;  // unused
    return false;
#endif
}
