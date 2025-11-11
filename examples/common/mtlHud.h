//
//   Copyright 2016 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_MTL_HUD_H
#define OPENSUBDIV_EXAMPLES_MTL_HUD_H

@protocol MTLTexture;
@protocol MTLBuffer;
@protocol MTLRenderCommandEncoder;
@protocol MTLRenderPipelineState;
@protocol MTLDevice;
@class MTLRenderPipelineDescriptor;
@class MTLDepthStencilDescriptor;

#include "hud.h"
#include "mtlUtils.h"

class MTLhud : public Hud {

public:
    MTLhud();
    ~MTLhud();

    virtual void Init(id<MTLDevice> device, MTLRenderPipelineDescriptor* parentPipelineDescriptor, MTLDepthStencilDescriptor* depthStencilStateDescriptor,
    				  int width, int height, int framebufferWidth, int framebufferHeight);

    virtual void Rebuild(int width, int height,
                         int framebufferWidth, int framebufferHeight);

    virtual bool Flush(id<MTLRenderCommandEncoder> encoder);

    id<MTLTexture> GetFontTexture() const {
        return _fontTexture;
    }

    void FillBackground(id<MTLRenderCommandEncoder> encoder);
    
    float UIScale = 1.0f;

private:
	id<MTLDevice> _device;
    id<MTLTexture> _fontTexture;
	OpenSubdiv::OPENSUBDIV_VERSION::Osd::MTLRingBuffer<float, 1> _staticBuffer;
	OpenSubdiv::OPENSUBDIV_VERSION::Osd::MTLRingBuffer<float, 3> _dynamicBuffer;

    id<MTLRenderPipelineState> _fgPipelineState, _bgPipelineState;
    id<MTLDepthStencilState> _depthStencilState;
};


#endif //OPENSUBDIV_EXAMPLES_MTL_HUD_H
