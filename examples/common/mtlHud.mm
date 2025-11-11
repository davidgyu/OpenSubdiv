//
//   Copyright 2016 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "mtlHud.h"
#include "font_image.h"
#include "simple_math.h"
#include <Metal/Metal.h>

static const char* shaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
	float2 position [[attribute(0)]];
	float3 color [[attribute(1)]];
	float2 uv [[attribute(2)]];
};

struct VertexOutput {
	float4 position [[position]];
	float4 color;
	float2 uv; 
};



vertex VertexOutput fg_vertex(
	VertexInput in [[stage_in]],
	constant const float4x4& ModelViewProjectionMatrix [[buffer(1)]],
    constant const float& UIScale [[buffer(2)]]
	) {
	VertexOutput out;

	out.position = ModelViewProjectionMatrix * float4(in.position * UIScale, 0, 1);
	out.color = float4(in.color, 1);
	out.uv = in.uv;

	return out;
}

fragment float4 fg_fragment(
	VertexOutput in [[stage_in]],
	texture2d<float, access::sample> fontTexture [[texture(0)]]
	) {

    constexpr sampler s = sampler(coord::normalized, address::clamp_to_zero, filter::nearest);
	auto c = fontTexture.sample(s, in.uv);
	if(c.a == 0)
		discard_fragment();

	return c * in.color;
}

constant float4 bgVertices[] = {
	{-1, 1, 0, 1},
	{1, 1, 0, 1},
	{-1, -1, 0, 1},
	{1, -1, 0, 1}
};

vertex VertexOutput bg_vertex(uint vertex_id [[vertex_id]]) {
	VertexOutput out;

	out.position = bgVertices[vertex_id];
	out.uv = bgVertices[vertex_id].xy * 0.5 + 0.5 * 3.14159;

	return out;
}

fragment float4 bg_fragment(VertexOutput in [[stage_in]]) {
	return float4(float3(mix(0.1, 0.5, sin(in.uv.y))), 1);
}

)";


MTLhud::MTLhud() {


}

MTLhud::~MTLhud() {

}

void
MTLhud::Init(id<MTLDevice> device, MTLRenderPipelineDescriptor* parentPipelineDescriptor, MTLDepthStencilDescriptor* depthStencilStateDescriptor, int width, int height, int framebufferWidth, int framebufferHeight) {
	Hud::Init(width, height, framebufferWidth, framebufferHeight);

	@autoreleasepool {
		_device = device;
        const auto textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:FONT_TEXTURE_WIDTH height:FONT_TEXTURE_HEIGHT mipmapped:false];
		_fontTexture = [_device newTextureWithDescriptor:textureDescriptor];
		[_fontTexture replaceRegion: { {0, 0, 0}, {FONT_TEXTURE_WIDTH, FONT_TEXTURE_HEIGHT, 1} } mipmapLevel:0 withBytes:font_image bytesPerRow:4 * FONT_TEXTURE_WIDTH];

		NSError* err = nil;
        const auto library = [_device newLibraryWithSource:@(shaderSource) options:nil error:&err];
        assert(err == nil);

		const auto renderPipelineDescriptor = (MTLRenderPipelineDescriptor*)[parentPipelineDescriptor copy];
		renderPipelineDescriptor.vertexFunction = [library newFunctionWithName:@"bg_vertex"];
		renderPipelineDescriptor.fragmentFunction = [library newFunctionWithName:@"bg_fragment"];

		_bgPipelineState = [_device newRenderPipelineStateWithDescriptor:renderPipelineDescriptor error:&err];

		renderPipelineDescriptor.vertexFunction = [library newFunctionWithName:@"fg_vertex"];
		renderPipelineDescriptor.fragmentFunction = [library newFunctionWithName:@"fg_fragment"];
		const auto vertexDescriptor = renderPipelineDescriptor.vertexDescriptor;

		vertexDescriptor.layouts[0].stride = sizeof(float) * 7;
		vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
		vertexDescriptor.layouts[0].stepRate = 1;

		vertexDescriptor.attributes[0].offset = 0;
		vertexDescriptor.attributes[0].format = MTLVertexFormatFloat2;
		vertexDescriptor.attributes[0].bufferIndex = 0;

		vertexDescriptor.attributes[1].offset = sizeof(float) * 2;
		vertexDescriptor.attributes[1].format = MTLVertexFormatFloat3;
		vertexDescriptor.attributes[1].bufferIndex = 0;

		vertexDescriptor.attributes[2].offset = sizeof(float) * 5;
		vertexDescriptor.attributes[2].format = MTLVertexFormatFloat2;
		vertexDescriptor.attributes[2].bufferIndex = 0;

		_fgPipelineState = [_device newRenderPipelineStateWithDescriptor:renderPipelineDescriptor error:&err];

		_depthStencilState = [_device newDepthStencilStateWithDescriptor:depthStencilStateDescriptor];
	}

}

void 
MTLhud::Rebuild(int width, int height, int framebufferWidth, int framebufferHeight) {
	Hud::Rebuild(width, height, framebufferWidth, framebufferHeight);

	_staticBuffer.alloc(_device, getStaticVboSource().size(), @"MTLhud static buffer");
	std::copy(getStaticVboSource().begin(), getStaticVboSource().end(), _staticBuffer.data());
	_staticBuffer.markModified();
}

bool
MTLhud::Flush(id<MTLRenderCommandEncoder> encoder) {
	if(!Hud::Flush())
		return false;

	if(_dynamicBuffer.buffer().length < sizeof(float) * getVboSource().size()) {
		_dynamicBuffer.alloc(_device, getVboSource().size(), @"MTLhud dynamic buffer");
	} else {
		_dynamicBuffer.next();
	}

	std::copy(getVboSource().begin(), getVboSource().end(), _dynamicBuffer.data());
	_dynamicBuffer.markModified();
	const auto numVertices = getVboSource().size() / 7;
	getVboSource().clear();

    float proj[16];
    ortho(proj, 0, 0, float(GetWidth()), float(GetHeight()));

	[encoder setVertexBuffer: _dynamicBuffer.buffer() offset:0 atIndex:0];
	[encoder setRenderPipelineState: _fgPipelineState];
	[encoder setDepthStencilState: _depthStencilState];
	[encoder setVertexBytes: proj length: sizeof(proj) atIndex:1];
    [encoder setVertexBytes: &UIScale length:sizeof(UIScale) atIndex:2];
	[encoder setFragmentTexture: _fontTexture atIndex:0];
    
    if(numVertices > 0) {
        [encoder setVertexBuffer:_dynamicBuffer.buffer() offset:0 atIndex:0];
        [encoder drawPrimitives: MTLPrimitiveTypeTriangle vertexStart:0 vertexCount: numVertices];
    }
    
    auto numStaticVertices = [_staticBuffer.buffer() length] / (7 * sizeof(float));
    if(numStaticVertices > 0) {
        [encoder setVertexBuffer:_staticBuffer.buffer() offset:0 atIndex:0];
        [encoder drawPrimitives: MTLPrimitiveTypeTriangle vertexStart:0 vertexCount: numStaticVertices];
    }

	return true;
}

void
MTLhud::FillBackground(id<MTLRenderCommandEncoder> encoder) {
	[encoder setRenderPipelineState: _bgPipelineState];
	[encoder setDepthStencilState: _depthStencilState];
	[encoder drawPrimitives: MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount: 4];
}
