//
//   Copyright 2015 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef enum {
    kEndCapNone = 0,
    kEndCapBSplineBasis,
    kEndCapGregoryBasis,
    kEndCapLegacyGregory,
} EndCap;

typedef enum {
    kFVarLinearNone = 0,
    kFVarLinearCornersOnly,
    kFVarLinearCornersPlus1,
    kFVarLinearCornersPlus2,
    kFVarLinearBoundaries,
    kFVarLinearAll
} FVarBoundary;

typedef enum {
    kCPU = 0,
    kMetal,
} KernelType;

typedef enum {
	kShadingUV = 0,
	kShadingVARYING,
	kShadingNORMAL,
	kShadingSHADE,
	kShadingFACEVARYING,
	kShadingMEAN_CURVATURE
} ShadingMode;

typedef struct {
    float rotationX;
    float rotationY;
    float dollyDistance;
    float aspectRatio;
} Camera;

@class OSDRenderer;

@protocol OSDRendererDelegate <NSObject>
-(id<MTLDevice>)deviceFor:(OSDRenderer*)renderer;
-(id<MTLCommandQueue>)commandQueueFor:(OSDRenderer*)renderer;
-(MTLRenderPassDescriptor*)renderPassDescriptorFor:(OSDRenderer*)renderer;
-(void)setupDepthStencilState:(MTLDepthStencilDescriptor*)descriptor for:(OSDRenderer*)renderer;
-(void)setupRenderPipelineState:(MTLRenderPipelineDescriptor*)descriptor for:(OSDRenderer*)renderer;
@end

@interface OSDRenderer : NSObject

-(instancetype)initWithDelegate:(id<OSDRendererDelegate>)delegate;

-(id<MTLRenderCommandEncoder>)drawFrame:(id<MTLCommandBuffer>)commandBuffer
                              frameBeginTimestamp:(double)frameBeginTimestamp;

@property (readonly, nonatomic) id<OSDRendererDelegate> delegate;

@property (nonatomic) unsigned refinementLevel;
@property (nonatomic) float tessellationLevel;

@property (readonly, nonatomic) NSArray<NSString*>* loadedModels;
@property (nonatomic) NSString* currentModel;

@property (readonly, nonatomic) Camera* camera;

@property (readonly, nonatomic) int* patchCounts;

@property (nonatomic) bool useSingleCrease;
@property (nonatomic) bool useInfinitelySharpPatch;
@property (nonatomic) bool useRandomStart;
@property (nonatomic) bool useAnimateParticles;
@property (nonatomic) bool useStageIn;
@property (nonatomic) bool useAdaptive;
@property (nonatomic) bool freeze;
@property (nonatomic) bool animateVertices;
@property (nonatomic) bool displayControlMeshEdges;
@property (nonatomic) bool displayControlMeshVertices;
@property (nonatomic) ShadingMode shadingMode;
@property (nonatomic) EndCap endCapMode;
@property (nonatomic) FVarBoundary fVarBoundary;
@property (nonatomic) KernelType kernelType;

@end
