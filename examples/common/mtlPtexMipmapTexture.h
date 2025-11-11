//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_MTL_PTEX_MIPMAP_TEXTURE_H
#define OPENSUBDIV_EXAMPLES_MTL_PTEX_MIPMAP_TEXTURE_H

#include <opensubdiv/osd/mtlCommon.h>
#include <opensubdiv/osd/nonCopyable.h>
#include <Ptexture.h>


@protocol MTLBuffer;
@protocol MTLTexture;
@class MTLTextureDescriptor;

class MTLPtexMipmapTexture : OpenSubdiv::Osd::NonCopyable<MTLPtexMipmapTexture> {
public:
    static MTLPtexMipmapTexture * Create(OpenSubdiv::Osd::MTLContext * deviceContext,
                                        PtexTexture * reader,
                                        int maxLevels = 10);
    
    
    static const char* GetShaderSource();
    
    id<MTLBuffer> GetLayoutBuffer() const { return _layout; }
    id<MTLTexture> GetTexelsTexture() const { return _texels; }
    
private:
    MTLPtexMipmapTexture();
    
    id<MTLBuffer> _layout;
    id<MTLTexture> _texels;

    MTLTextureDescriptor* _textureDescriptor;
};

#endif  // OPENSUBDIV_EXAMPLES_MTL_PTEX_TEXTURE_H
