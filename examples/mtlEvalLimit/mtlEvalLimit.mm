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

#import "mtlEvalLimit.h"

#import <simd/simd.h>
#import <algorithm>
#import <cfloat>
#import <fstream>
#import <iostream>
#import <iterator>
#import <string>
#import <sstream>
#import <vector>
#import <memory>

#import <far/error.h>
#import <far/topologyRefiner.h>
#import <far/patchTableFactory.h>
#import <far/stencilTableFactory.h>
#import <osd/mesh.h>
#import <osd/cpuVertexBuffer.h>
#import <osd/cpuEvaluator.h>
#import <osd/cpuPatchTable.h>
#import <osd/mtlVertexBuffer.h>
#import <osd/mtlMesh.h>
#import <osd/mtlPatchTable.h>
#import <osd/mtlComputeEvaluator.h>
#import <osd/mtlPatchShaderSource.h>

#import "../common/simple_math.h"
#import "../../regression/common/far_utils.h"
#import "init_shapes.h"
#import "particles.h"
#import "../common/mtlUtils.h"
#import "../common/mtlControlMeshDisplay.h"

#define FRAME_CONST_BUFFER_INDEX 0
#define DRAWMODE_BUFFER_INDEX 1
#define VERTEX_BUFFER_INDEX 2
#define DERIV1_BUFFER_INDEX 3
#define DERIV2_BUFFER_INDEX 4
#define PATCHCOORD_BUFFER_INDEX 5
#define FVARDATA_BUFFER_INDEX 6

using namespace OpenSubdiv::OPENSUBDIV_VERSION;

template <> Far::StencilTable const * Osd::convertToCompatibleStencilTable<OpenSubdiv::Far::StencilTable,
                                                                           OpenSubdiv::Far::StencilTable,
                                                                           OpenSubdiv::Osd::MTLContext>(
    OpenSubdiv::Far::StencilTable const *table, OpenSubdiv::Osd::MTLContext*  /*context*/) {
    // no need for conversion
    // XXX: We don't want to even copy.
    if (not table) return NULL;
    return new Far::StencilTable(*table);
}

using CPUMeshType = Osd::Mesh<
    Osd::CPUMTLVertexBuffer,
    Far::StencilTable,
    Osd::CpuEvaluator,
    Osd::MTLPatchTable,
    Osd::MTLContext>;

using mtlMeshType = Osd::Mesh<
    Osd::CPUMTLVertexBuffer,
    Osd::MTLStencilTable,
    Osd::MTLComputeEvaluator,
    Osd::MTLPatchTable,
    Osd::MTLContext>;

using MTLMeshInterface = Osd::MTLMeshInterface;

struct alignas(16) PerFrameConstants
{
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
    float ModelViewInverseMatrix[16];
};

static const char* shaderSource =
#include "mtlEvalLimit.gen.h"
;

// input and output vertex data
class EvalOutputBase {
public:
    virtual ~EvalOutputBase() {}
    virtual id<MTLBuffer> BindSourceData(Osd::MTLContext *context) const = 0;
    virtual id<MTLBuffer> BindVertexData(Osd::MTLContext *context) const = 0;
    virtual id<MTLBuffer> Bind1stDerivatives(Osd::MTLContext *context) const = 0;
    virtual id<MTLBuffer> Bind2ndDerivatives(Osd::MTLContext *context) const = 0;
    virtual id<MTLBuffer> BindFaceVaryingData(Osd::MTLContext *context) const = 0;
    virtual id<MTLBuffer> BindPatchCoords(Osd::MTLContext *context) const = 0;
    virtual void UpdateData(const float *src, int startVertex, int numVertices) = 0;
    virtual void UpdateVaryingData(const float *src, int startVertex, int numVertices) = 0;
    virtual void UpdateFaceVaryingData(const float *src, int startVertex, int numVertices) = 0;
    virtual bool HasFaceVaryingData() const = 0;
    virtual void Refine() = 0;
    virtual void EvalPatches() = 0;
    virtual void EvalPatchesWith1stDerivatives() = 0;
    virtual void EvalPatchesWith2ndDerivatives() = 0;
    virtual void EvalPatchesVarying() = 0;
    virtual void EvalPatchesFaceVarying() = 0;
    virtual void UpdatePatchCoords(
        std::vector<Osd::PatchCoord> const &patchCoords) = 0;
};

// note: Since we don't have a class for device-patchcoord container in osd,
// we cheat to use vertexbuffer as a patch-coord (5int) container.
//
// Please don't follow the pattern in your actual application.
//
template<typename SRC_VERTEX_BUFFER, typename EVAL_VERTEX_BUFFER,
         typename STENCIL_TABLE, typename PATCH_TABLE, typename EVALUATOR,
         typename DEVICE_CONTEXT = void>
class EvalOutput : public EvalOutputBase {
public:
    typedef OpenSubdiv::Osd::EvaluatorCacheT<EVALUATOR> EvaluatorCache;

    EvalOutput(Far::StencilTable const *vertexStencils,
               Far::StencilTable const *varyingStencils,
               Far::StencilTable const *faceVaryingStencils,
               int fvarChannel, int fvarWidth,
               int numParticles, Far::PatchTable const *patchTable,
               EvaluatorCache *evaluatorCache = NULL,
               DEVICE_CONTEXT *deviceContext = NULL)
        : _srcDesc(       /*offset*/ 0, /*length*/ 3, /*stride*/ 3),
          _srcVaryingDesc(/*offset*/ 0, /*length*/ 3, /*stride*/ 3),
          _srcFVarDesc(   /*offset*/ 0, /*length*/ fvarWidth, /*stride*/ fvarWidth),
          _vertexDesc(    /*offset*/ 0, /*length*/ 3, /*stride*/ 6),
          _varyingDesc(   /*offset*/ 3, /*length*/ 3, /*stride*/ 6),
          _fvarDesc(      /*offset*/ 0, /*length*/ fvarWidth, /*stride*/ fvarWidth),
          _duDesc(        /*offset*/ 0, /*length*/ 3, /*stride*/ 6),
          _dvDesc(        /*offset*/ 3, /*length*/ 3, /*stride*/ 6),
          _duuDesc(       /*offset*/ 0, /*length*/ 3, /*stride*/ 9),
          _duvDesc(       /*offset*/ 3, /*length*/ 3, /*stride*/ 9),
          _dvvDesc(       /*offset*/ 6, /*length*/ 3, /*stride*/ 9),
          _deviceContext(deviceContext) {

        // total number of vertices = coarse points + refined points + local points
        int numTotalVerts = vertexStencils->GetNumControlVertices()
                          + vertexStencils->GetNumStencils();

        _srcData = SRC_VERTEX_BUFFER::Create(3, numTotalVerts, _deviceContext);
        _srcVaryingData = SRC_VERTEX_BUFFER::Create(3, numTotalVerts, _deviceContext);
        _vertexData = EVAL_VERTEX_BUFFER::Create(6, numParticles, _deviceContext);
        _deriv1 = EVAL_VERTEX_BUFFER::Create(6, numParticles, _deviceContext);
        _deriv2 = EVAL_VERTEX_BUFFER::Create(9, numParticles, _deviceContext);
        _patchTable = PATCH_TABLE::Create(patchTable, _deviceContext);
        _patchCoords = NULL;
        _numCoarseVerts = vertexStencils->GetNumControlVertices();
        _vertexStencils =
            Osd::convertToCompatibleStencilTable<STENCIL_TABLE>(vertexStencils, _deviceContext);
        _varyingStencils =
            Osd::convertToCompatibleStencilTable<STENCIL_TABLE>(varyingStencils, _deviceContext);

        if (faceVaryingStencils) {
            _numCoarseFVarVerts = faceVaryingStencils->GetNumControlVertices();
            int numTotalFVarVerts = faceVaryingStencils->GetNumControlVertices()
                                  + faceVaryingStencils->GetNumStencils();
            _srcFVarData = EVAL_VERTEX_BUFFER::Create(2, numTotalFVarVerts, _deviceContext);
            _fvarData = EVAL_VERTEX_BUFFER::Create(fvarWidth, numParticles, _deviceContext);
            _faceVaryingStencils =
                Osd::convertToCompatibleStencilTable<STENCIL_TABLE>(faceVaryingStencils, _deviceContext);
            _fvarChannel = fvarChannel;
            _fvarWidth = fvarWidth;
        } else {
            _numCoarseFVarVerts = 0;
            _srcFVarData = NULL;
            _fvarData = NULL;
            _faceVaryingStencils = NULL;
            _fvarChannel = 0;
            _fvarWidth = 0;
        }
        _evaluatorCache = evaluatorCache;
    }
    ~EvalOutput() {
        delete _srcData;
        delete _srcVaryingData;
        delete _srcFVarData;
        delete _vertexData;
        delete _deriv1;
        delete _deriv2;
        delete _fvarData;
        delete _patchTable;
        delete _patchCoords;
        delete _vertexStencils;
        delete _varyingStencils;
        delete _faceVaryingStencils;
    }
    virtual id<MTLBuffer> BindSourceData(Osd::MTLContext *context) const {
        return _srcData->BindMTLBuffer(context);
    }
    virtual id<MTLBuffer> BindVertexData(Osd::MTLContext *context) const {
        return _vertexData->BindMTLBuffer(context);
    }
    virtual id<MTLBuffer> Bind1stDerivatives(Osd::MTLContext *context) const {
        return _deriv1->BindMTLBuffer(context);
    }
    virtual id<MTLBuffer> Bind2ndDerivatives(Osd::MTLContext *context) const {
        return _deriv2->BindMTLBuffer(context);
    }
    virtual id<MTLBuffer> BindFaceVaryingData(Osd::MTLContext *context) const {
        return _fvarData->BindMTLBuffer(context);
    }
    virtual id<MTLBuffer> BindPatchCoords(Osd::MTLContext *context) const {
        return _patchCoords->BindMTLBuffer(context);
    }
    virtual void UpdateData(const float *src, int startVertex, int numVertices) {
        _srcData->UpdateData(src, startVertex, numVertices, _deviceContext);
    }
    virtual void UpdateVaryingData(const float *src, int startVertex, int numVertices) {
        _srcVaryingData->UpdateData(src, startVertex, numVertices, _deviceContext);
    }
    virtual void UpdateFaceVaryingData(const float *src, int startVertex, int numVertices) {
        _srcFVarData->UpdateData(src, startVertex, numVertices, _deviceContext);
    }
    virtual bool HasFaceVaryingData() const {
        return _faceVaryingStencils != NULL;
    }
    virtual void Refine() {
        Osd::BufferDescriptor dstDesc = _srcDesc;
        dstDesc.offset += _numCoarseVerts * _srcDesc.stride;

        EVALUATOR *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcDesc, dstDesc, _deviceContext);

        EVALUATOR::EvalStencils(_srcData, _srcDesc,
                                _srcData, dstDesc,
                                _vertexStencils,
                                evalInstance,
                                _deviceContext);

        dstDesc = _srcVaryingDesc;
        dstDesc.offset += _numCoarseVerts * _srcVaryingDesc.stride;
        evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcVaryingDesc, dstDesc, _deviceContext);

        EVALUATOR::EvalStencils(_srcVaryingData, _srcVaryingDesc,
                                _srcVaryingData, dstDesc,
                                _varyingStencils,
                                evalInstance,
                                _deviceContext);

        if (HasFaceVaryingData()) {
            Osd::BufferDescriptor dstFVarDesc = _srcFVarDesc;
            dstFVarDesc.offset += _numCoarseFVarVerts * _srcFVarDesc.stride;

            evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
                _evaluatorCache, _srcFVarDesc, dstFVarDesc, _deviceContext);

            EVALUATOR::EvalStencils(_srcFVarData, _srcFVarDesc,
                                    _srcFVarData, dstFVarDesc,
                                    _faceVaryingStencils,
                                    evalInstance,
                                    _deviceContext);
        }

    }
    virtual void EvalPatches() {
        EVALUATOR *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcDesc, _vertexDesc, _deviceContext);

        EVALUATOR::EvalPatches(
            _srcData, _srcDesc,
            _vertexData, _vertexDesc,
            _patchCoords->GetNumVertices(),
            _patchCoords,
            _patchTable, evalInstance, _deviceContext);
    }
    virtual void EvalPatchesWith1stDerivatives() {
        EVALUATOR *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcDesc, _vertexDesc, _duDesc, _dvDesc, _deviceContext);
        EVALUATOR::EvalPatches(
            _srcData, _srcDesc,
            _vertexData, _vertexDesc,
            _deriv1, _duDesc,
            _deriv1, _dvDesc,
            _patchCoords->GetNumVertices(),
            _patchCoords,
            _patchTable, evalInstance, _deviceContext);
    }
    virtual void EvalPatchesWith2ndDerivatives() {
        EVALUATOR *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcDesc, _vertexDesc,
            _duDesc, _dvDesc, _duuDesc, _duvDesc, _dvvDesc,
            _deviceContext);
        EVALUATOR::EvalPatches(
            _srcData, _srcDesc,
            _vertexData, _vertexDesc,
            _deriv1, _duDesc,
            _deriv1, _dvDesc,
            _deriv2, _duuDesc,
            _deriv2, _duvDesc,
            _deriv2, _dvvDesc,
            _patchCoords->GetNumVertices(),
            _patchCoords,
            _patchTable, evalInstance, _deviceContext);
    }
    virtual void EvalPatchesVarying() {
        EVALUATOR *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcVaryingDesc, _varyingDesc, _deviceContext);

        EVALUATOR::EvalPatchesVarying(
            _srcVaryingData, _srcVaryingDesc,
            // varying data is interleaved in vertexData.
            _vertexData, _varyingDesc,
            _patchCoords->GetNumVertices(),
            _patchCoords,
            _patchTable, evalInstance, _deviceContext);
    }
    virtual void EvalPatchesFaceVarying() {
        EVALUATOR *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcFVarDesc, _fvarDesc, _deviceContext);

        EVALUATOR::EvalPatchesFaceVarying(
            _srcFVarData, _srcFVarDesc,
            _fvarData, _fvarDesc,
            _patchCoords->GetNumVertices(),
            _patchCoords,
            _patchTable, _fvarChannel, evalInstance, _deviceContext);
    }
    virtual void UpdatePatchCoords(
        std::vector<Osd::PatchCoord> const &patchCoords) {
        if (_patchCoords &&
            _patchCoords->GetNumVertices() != (int)patchCoords.size()) {
            delete _patchCoords;
            _patchCoords = NULL;
        }
        if (! _patchCoords) {
            _patchCoords = EVAL_VERTEX_BUFFER::Create(5,
                                                      (int)patchCoords.size(),
                                                      _deviceContext);
        }
        _patchCoords->UpdateData((float*)&patchCoords[0], 0, (int)patchCoords.size(), _deviceContext);
    }
private:
    SRC_VERTEX_BUFFER *_srcData;
    SRC_VERTEX_BUFFER *_srcVaryingData;
    EVAL_VERTEX_BUFFER *_srcFVarData;
    EVAL_VERTEX_BUFFER *_vertexData;
    EVAL_VERTEX_BUFFER *_deriv1;
    EVAL_VERTEX_BUFFER *_deriv2;
    EVAL_VERTEX_BUFFER *_fvarData;
    EVAL_VERTEX_BUFFER *_patchCoords;
    PATCH_TABLE *_patchTable;
    Osd::BufferDescriptor _srcDesc;
    Osd::BufferDescriptor _srcVaryingDesc;
    Osd::BufferDescriptor _srcFVarDesc;
    Osd::BufferDescriptor _vertexDesc;
    Osd::BufferDescriptor _varyingDesc;
    Osd::BufferDescriptor _fvarDesc;
    Osd::BufferDescriptor _duDesc;
    Osd::BufferDescriptor _dvDesc;
    Osd::BufferDescriptor _duuDesc;
    Osd::BufferDescriptor _duvDesc;
    Osd::BufferDescriptor _dvvDesc;
    int _numCoarseVerts;
    int _numCoarseFVarVerts;

    STENCIL_TABLE const *_vertexStencils;
    STENCIL_TABLE const *_varyingStencils;
    STENCIL_TABLE const *_faceVaryingStencils;

    int _fvarChannel;
    int _fvarWidth;

    EvaluatorCache *_evaluatorCache;
    DEVICE_CONTEXT *_deviceContext;
};

static void
createRandomColors(int nverts, int stride, float * colors) {

    // large Pell prime number
    srand( static_cast<int>(2147483647) );

    for (int i=0; i<nverts; ++i) {
        colors[i*stride+0] = (float)rand()/(float)RAND_MAX;
        colors[i*stride+1] = (float)rand()/(float)RAND_MAX;
        colors[i*stride+2] = (float)rand()/(float)RAND_MAX;
    }
}

using Osd::MTLRingBuffer;

#define FRAME_LAG 3
template<typename DataType>
using PerFrameBuffer = MTLRingBuffer<DataType, FRAME_LAG>;

@implementation OSDRenderer {

    PerFrameBuffer<PerFrameConstants> _frameConstantsBuffer;
    
    id<MTLRenderPipelineState> _renderPipeline;
    id<MTLDepthStencilState> _readWriteDepthStencilState;
    id<MTLDepthStencilState> _readOnlyDepthStencilState;

    Camera _cameraData;
    Osd::MTLContext _context;
    
    int _numVertexElements;
    int _numVaryingElements;
    int _numFaceVaryingElements;
    int _numVertices;
    int _frameCount;
    int _animationFrames;
    std::vector<float> _vertexData, _animatedVertices, _varyingColors;
    std::unique_ptr<MTLControlMeshDisplay> _controlMesh;
    std::unique_ptr<Shape> _shape;

    std::unique_ptr<Far::PatchTable> _patchTable;
    std::unique_ptr<EvalOutputBase> _evalMesh;
    std::unique_ptr<STParticles> _particles;
    int _numParticles;

    bool _needsRebuild;
    simd::float3 _meshCenter;
    NSMutableArray<NSString*>* _loadedModels;

    float _currentTime;
    float _prevTime;
}

-(Camera*)camera {
    return &_cameraData;
}

-(int *)patchCounts {
    return 0;
}

-(instancetype)initWithDelegate:(id<OSDRendererDelegate>)delegate {
    self = [super init];
    if(self) {
        self.useSingleCrease = true;
        self.useStageIn = !TARGET_OS_EMBEDDED;
        self.endCapMode = kEndCapBSplineBasis;
        self.useRandomStart = true;
        self.useAnimateParticles = true;
        self.kernelType = kMetal;
        self.refinementLevel = 2;
        self.tessellationLevel = 8;
        self.shadingMode = kShadingUV;
        
        _frameCount = 0;
        _animationFrames = 0;
        _delegate = delegate;
        _context.device = [delegate deviceFor:self];
        _context.commandQueue = [delegate commandQueueFor:self];
        
        _needsRebuild = true;

        _currentTime = _prevTime = 0;
        
        [self _initializeBuffers];
        [self _initializeCamera];
        [self _initializeModels];
    }
    return self;
}

-(id<MTLRenderCommandEncoder>)drawFrame:(id<MTLCommandBuffer>)commandBuffer
                              frameBeginTimestamp:(double)frameBeginTimestamp {
    if(_needsRebuild) {
        [self _rebuildState];
    }

    _currentTime = (float)frameBeginTimestamp;

    if(!_freeze) {
        if(_animateVertices) {
            _animatedVertices.resize(_vertexData.size());
            auto p = _vertexData.data();
            auto n = _animatedVertices.data();
            
            int numElements = _numVertexElements + _numVaryingElements;

            float r = sin(_animationFrames*0.01f) * _animateVertices;
            for (int i = 0; i < _numVertices; ++i) {
                float move = 0.05f*cosf(p[0]*20+_animationFrames*0.01f);
                float ct = cos(p[2] * r);
                float st = sin(p[2] * r);
                n[0] = p[0]*ct + p[1]*st;
                n[1] = -p[0]*st + p[1]*ct;
                n[2] = p[2];
                
                for (int j = 0; j < _numVaryingElements; ++j) {
                    n[3 + j] = p[3 + j];
                } 

                p += numElements;
                n += numElements;
            }

            _animationFrames++;

            _evalMesh->UpdateData(_animatedVertices.data(), 0, _numVertices);
        }
    }

    _evalMesh->Refine();

    if (_shadingMode == kShadingVARYING) {
        _evalMesh->UpdateVaryingData(_varyingColors.data(), 0, _numVertices);
    }

    float elapsed = _currentTime - _prevTime;
    _particles->Update(elapsed);
    _prevTime = _currentTime;

    // update patchcoord to be evaluated
    std::vector<OpenSubdiv::Osd::PatchCoord> const &patchCoords = _particles->GetPatchCoords();
    _evalMesh->UpdatePatchCoords(patchCoords);

    // Evaluate the positions of the samples on the limit surface
    if (_shadingMode == kShadingMEAN_CURVATURE) {
        // evaluate positions and 2nd derivatives
        _evalMesh->EvalPatchesWith2ndDerivatives();
    } else if (_shadingMode == kShadingNORMAL || _shadingMode == kShadingSHADE) {
        // evaluate positions and 1st derivatives
        _evalMesh->EvalPatchesWith1stDerivatives();
    } else {
        // evaluate positions
        _evalMesh->EvalPatches();
    }

    // color
    if (_shadingMode == kShadingVARYING) {
        _evalMesh->EvalPatchesVarying();
    } else if (_shadingMode == kShadingFACEVARYING && _evalMesh->HasFaceVaryingData()) {
        _evalMesh->EvalPatchesFaceVarying();
    }

    [self _updateState];
    
    auto renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:[_delegate renderPassDescriptorFor: self]];
    [self _renderMesh:renderEncoder];

    _frameConstantsBuffer.next();
    
    _frameCount++;
    
    return renderEncoder;
}

-(void)_renderMesh:(id<MTLRenderCommandEncoder>)renderCommandEncoder {
    auto vertexBuffer = _evalMesh->BindVertexData(&_context);
    auto deriv1Buffer = _evalMesh->Bind1stDerivatives(&_context);
    auto deriv2Buffer = _evalMesh->Bind2ndDerivatives(&_context);
    auto coordsBuffer = _evalMesh->BindPatchCoords(&_context);
    auto fvarBuffer = _evalMesh->BindFaceVaryingData(&_context);
    
    [renderCommandEncoder setVertexBuffer:vertexBuffer offset:0 atIndex:VERTEX_BUFFER_INDEX];
    [renderCommandEncoder setVertexBuffer:deriv1Buffer offset:0 atIndex:DERIV1_BUFFER_INDEX];
    [renderCommandEncoder setVertexBuffer:deriv2Buffer offset:0 atIndex:DERIV2_BUFFER_INDEX];
    [renderCommandEncoder setVertexBuffer:coordsBuffer offset:0 atIndex:PATCHCOORD_BUFFER_INDEX];
    [renderCommandEncoder setVertexBuffer:fvarBuffer offset:0 atIndex:FVARDATA_BUFFER_INDEX];

    [renderCommandEncoder setVertexBytes:&_shadingMode length:sizeof(_shadingMode) atIndex:DRAWMODE_BUFFER_INDEX];

    [renderCommandEncoder setVertexBuffer:_frameConstantsBuffer offset:0 atIndex:FRAME_CONST_BUFFER_INDEX];
    
	[renderCommandEncoder setDepthStencilState:_readWriteDepthStencilState];
	[renderCommandEncoder setRenderPipelineState:_renderPipeline];

    [renderCommandEncoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:_numParticles];

    if (_displayControlMeshEdges) {
        [renderCommandEncoder setDepthStencilState:_readOnlyDepthStencilState]; _controlMesh->Draw(renderCommandEncoder, _evalMesh->BindSourceData(&_context), _frameConstantsBuffer->ModelViewProjectionMatrix);
    }
}

-(void)_rebuildState {
    [self _rebuildModel];
    [self _rebuildBuffers];
    [self _rebuildPipelines];
    
    _needsRebuild = false;
}

-(void)_rebuildModel {
    
    using namespace OpenSubdiv;
    using namespace Sdc;
    using namespace Osd;
    using namespace Far;
    auto shapeDesc = &g_defaultShapes[[_loadedModels indexOfObject:_currentModel]];
    _shape.reset(Shape::parseObj(shapeDesc->data.c_str(), shapeDesc->scheme));
    const auto scheme = shapeDesc->scheme;
    
    // create Far mesh (topology)
    Sdc::SchemeType sdctype = GetSdcType(*_shape);
    Sdc::Options sdcoptions = GetSdcOptions(*_shape);

    sdcoptions.SetFVarLinearInterpolation((OpenSubdiv::Sdc::Options::FVarLinearInterpolation)_fVarBoundary);

    std::unique_ptr<OpenSubdiv::Far::TopologyRefiner> refiner;
    refiner.reset(Far::TopologyRefinerFactory<Shape>::Create(*_shape,
                      Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions)));
    
    // save coarse topology (used for coarse mesh drawing)
    Far::TopologyLevel const & refBaseLevel = refiner->GetLevel(0);
    _numVertices = refBaseLevel.GetNumVertices();
    _numVertexElements = 3;
    _numVaryingElements = (_shadingMode == kShadingVARYING) ? 3 : 0;

    Far::StencilTable const * vertexStencils = NULL;
    Far::StencilTable const * varyingStencils = NULL;
    Far::StencilTable const * faceVaryingStencils = NULL;

    int fvarChannel = 0;
    int fvarWidth = _shape->GetFVarWidth();
    bool hasFVarData = !_shape->uvs.empty();

    {
        bool adaptive = (scheme == kCatmark);
        bool doSingleCreasePatch = (false && _useSingleCrease && scheme == kCatmark);
        bool doInfSharpPatch = (_useInfinitelySharpPatch && scheme == kCatmark);

        if (adaptive) {
            // Apply feature adaptive refinement to the mesh so that we can use the
            // limit evaluation API features.
            Far::TopologyRefiner::AdaptiveOptions options(_refinementLevel);
            options.considerFVarChannels = hasFVarData;
            options.useInfSharpPatch = doInfSharpPatch;
            refiner->RefineAdaptive(options);
        } else {
            Far::TopologyRefiner::UniformOptions options(_refinementLevel);
            refiner->RefineUniform(options);
        }

        // Generate stencil table to update the bi-cubic patches control
        // vertices after they have been re-posed (both for vertex & varying
        // interpolation)
        Far::StencilTableFactory::Options soptions;
        soptions.generateOffsets=true;
        soptions.generateIntermediateLevels=adaptive;

        vertexStencils =
            Far::StencilTableFactory::Create(*refiner, soptions);

        soptions.interpolationMode = Far::StencilTableFactory::INTERPOLATE_VARYING;
        varyingStencils =
            Far::StencilTableFactory::Create(*refiner, soptions);

        if (hasFVarData) {
            soptions.interpolationMode = Far::StencilTableFactory::INTERPOLATE_FACE_VARYING;
            soptions.fvarChannel = fvarChannel;
            faceVaryingStencils =
                Far::StencilTableFactory::Create(*refiner, soptions);
        }

        // Generate bi-cubic patch table for the limit surface
        Far::PatchTableFactory::Options poptions(_refinementLevel);
        if (_endCapMode == kEndCapBSplineBasis) {
            poptions.SetEndCapType(
                Far::PatchTableFactory::Options::ENDCAP_BSPLINE_BASIS);
        } else {
            poptions.SetEndCapType(
                Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);
        }
        poptions.useInfSharpPatch = doInfSharpPatch;
        poptions.generateFVarTables = hasFVarData;
        poptions.generateFVarLegacyLinearPatches = false;

        Far::PatchTable * patchTable =
            Far::PatchTableFactory::Create(*refiner, poptions);

        // append local points stencils
        if (Far::StencilTable const *localPointStencilTable =
            patchTable->GetLocalPointStencilTable()) {
            Far::StencilTable const *table =
                Far::StencilTableFactory::AppendLocalPointStencilTable(
                    *refiner, vertexStencils, localPointStencilTable);
            delete vertexStencils;
            vertexStencils = table;
        }
        if (Far::StencilTable const *localPointVaryingStencilTable =
            patchTable->GetLocalPointVaryingStencilTable()) {
            Far::StencilTable const *table =
                Far::StencilTableFactory::AppendLocalPointStencilTable(
                    *refiner,
                    varyingStencils, localPointVaryingStencilTable);
            delete varyingStencils;
            varyingStencils = table;
        }
        if (Far::StencilTable const *localPointFaceVaryingStencilTable =
            patchTable->GetLocalPointFaceVaryingStencilTable()) {
            Far::StencilTable const *table =
                Far::StencilTableFactory::AppendLocalPointStencilTableFaceVarying(
                    *refiner,
                    faceVaryingStencils, localPointFaceVaryingStencilTable);
            delete faceVaryingStencils;
            faceVaryingStencils = table;
        }

        _patchTable.reset(patchTable);
    }

    // Create the 'uv particles' manager - this class manages the limit
    // location samples (ptex face index, (s,t) and updates them between frames.
    // Note: the number of limit locations can be entirely arbitrary
    float speed = (_particles ? _particles->GetSpeed() : 0.2f) * 100;
    _numParticles = 65536;
    _particles.reset(new STParticles(*refiner, _patchTable.get(),
                                  _numParticles, !_useRandomStart));
    _numParticles = _particles->GetNumParticles();
    _particles->SetSpeed(speed);

    _prevTime = -1;
    _currentTime = 0;

    // In following template instantiations, same type of vertex buffers are
    // used for both source and destination (first and second template
    // parameters), since we'd like to draw control mesh wireframe too in
    // this example viewer.
    // If we don't need to draw the coarse control mesh, the src buffer doesn't
    // have to be interoperable to GL (it can be CpuVertexBuffer etc).

    int numElements = 3;
    if (_kernelType == kCPU) {
        static Osd::EvaluatorCacheT<Osd::CpuEvaluator> cpuEvaluatorCache;
        _evalMesh.reset(new EvalOutput<Osd::CPUMTLVertexBuffer,
                                       Osd::CPUMTLVertexBuffer,
                                       Far::StencilTable,
                                       Osd::CpuPatchTable,
                                       Osd::CpuEvaluator,
                                       MTLContext>
            (vertexStencils, varyingStencils, faceVaryingStencils,
             fvarChannel, fvarWidth,
             _numParticles, _patchTable.get(),
             &cpuEvaluatorCache, &_context));
    } else if (_kernelType == kMetal) {
        static Osd::EvaluatorCacheT<Osd::MTLComputeEvaluator> mtlEvaluatorCache;
        _evalMesh.reset(new EvalOutput<Osd::CPUMTLVertexBuffer,
                                       Osd::CPUMTLVertexBuffer,
                                       Osd::MTLStencilTable,
                                       Osd::MTLPatchTable,
                                       Osd::MTLComputeEvaluator,
                                       MTLContext>
            (vertexStencils, varyingStencils, faceVaryingStencils,
            fvarChannel, fvarWidth,
            _numParticles, _patchTable.get(),
            &mtlEvaluatorCache, &_context));
    }

    MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
    [_delegate setupRenderPipelineState:desc for:self];
    
    const auto vertexDescriptor = desc.vertexDescriptor;
    vertexDescriptor.layouts[0].stride = sizeof(float) * numElements;
    vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    vertexDescriptor.layouts[0].stepRate = 1;
    vertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
    vertexDescriptor.attributes[0].offset = 0;
    vertexDescriptor.attributes[0].bufferIndex = 0;
    
    _controlMesh.reset(new MTLControlMeshDisplay(_context.device, desc));
    _controlMesh->SetTopology(refBaseLevel);
    _controlMesh->SetEdgesDisplay(true);
    _controlMesh->SetVerticesDisplay(false);
    
    _vertexData.resize(refBaseLevel.GetNumVertices() * numElements);
    _meshCenter = simd::float3{0,0,0};
    
    for(int i = 0; i < refBaseLevel.GetNumVertices(); i++)
    {
        _vertexData[i * numElements + 0] = _shape->verts[i * 3 + 0];
        _vertexData[i * numElements + 1] = _shape->verts[i * 3 + 1];
        _vertexData[i * numElements + 2] = _shape->verts[i * 3 + 2];
    }
    
    for(auto vertexIdx = 0; vertexIdx < refBaseLevel.GetNumVertices(); vertexIdx++)
    {
        _meshCenter[0] += _vertexData[vertexIdx * numElements + 0];
        _meshCenter[1] += _vertexData[vertexIdx * numElements + 1];
        _meshCenter[2] += _vertexData[vertexIdx * numElements + 2];
    }
    
    _meshCenter /= (_shape->verts.size() / 3);

    // create random varying color
    {
        int numCoarseVerts = refiner->GetLevel(0).GetNumVertices();
        _varyingColors.resize(numCoarseVerts*3);
        createRandomColors(numCoarseVerts, 3, &_varyingColors[0]);
    }

    _evalMesh->UpdateData(_vertexData.data(), 0, refBaseLevel.GetNumVertices());

    if (_evalMesh->HasFaceVaryingData()) {
        _evalMesh->UpdateFaceVaryingData(
            &_shape->uvs[0], 0, (int)_shape->uvs.size()/_shape->GetFVarWidth());
    }

    refiner.release();
}

-(void)_updateState {
    [self _updateCamera];
    _frameConstantsBuffer.markModified();
}

-(void)_rebuildBuffers {
}

-(void)_rebuildPipelines {
    _renderPipeline = nil;
    
    std::stringstream shaderBuilder;
    shaderBuilder << shaderSource;
    const auto str = shaderBuilder.str();
    
    auto compileOptions = [[MTLCompileOptions alloc] init];
    auto preprocessor = [[NSMutableDictionary alloc] init];

#define DEFINE(x,y) preprocessor[@(#x)] = @(y)
    DEFINE(DRAWMODE_BUFFER_INDEX, DRAWMODE_BUFFER_INDEX);
    DEFINE(FRAME_CONST_BUFFER_INDEX, FRAME_CONST_BUFFER_INDEX);
    DEFINE(VERTEX_BUFFER_INDEX, VERTEX_BUFFER_INDEX);
    DEFINE(DERIV1_BUFFER_INDEX, DERIV1_BUFFER_INDEX);
    DEFINE(DERIV2_BUFFER_INDEX, DERIV2_BUFFER_INDEX);
    DEFINE(PATCHCOORD_BUFFER_INDEX, PATCHCOORD_BUFFER_INDEX);
    DEFINE(FVARDATA_BUFFER_INDEX, FVARDATA_BUFFER_INDEX);
#undef DEFINE

    compileOptions.preprocessorMacros = preprocessor;
    
    NSError* err = nil;
    auto librarySource = [NSString stringWithUTF8String:str.data()];
    auto library = [_context.device newLibraryWithSource:librarySource options:compileOptions error:&err];
    if(!library && err) {
        NSLog(@"%s", [err localizedDescription].UTF8String);
    }
    assert(library);
    auto vertexFunction = [library newFunctionWithName:@"vertex_main"];
    auto fragmentFunction = [library newFunctionWithName:@"fragment_main"];

    MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    [_delegate setupRenderPipelineState:pipelineDesc for:self];
    pipelineDesc.vertexFunction = vertexFunction;
    pipelineDesc.fragmentFunction = fragmentFunction;
    
    _renderPipeline = [_context.device newRenderPipelineStateWithDescriptor:pipelineDesc error:&err];
    if(!_renderPipeline && err)
    {
        NSLog(@"%s", [[err localizedDescription] UTF8String]);
    }
        
    MTLDepthStencilDescriptor* depthStencilDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthStencilDesc.depthCompareFunction = MTLCompareFunctionLess;
    
    [_delegate setupDepthStencilState:depthStencilDesc for:self];
    
    depthStencilDesc.depthWriteEnabled = YES;
    _readWriteDepthStencilState = [_context.device newDepthStencilStateWithDescriptor:depthStencilDesc];
    
    depthStencilDesc.depthWriteEnabled = NO;
    _readOnlyDepthStencilState = [_context.device newDepthStencilStateWithDescriptor:depthStencilDesc];
}

-(void)_updateCamera {
    auto pData = _frameConstantsBuffer.data();
    
    identity(pData->ModelViewMatrix);
    translate(pData->ModelViewMatrix, 0, 0, -_cameraData.dollyDistance);
    rotate(pData->ModelViewMatrix, _cameraData.rotationY, 1, 0, 0);
    rotate(pData->ModelViewMatrix, _cameraData.rotationX, 0, 1, 0);
    translate(pData->ModelViewMatrix, -_meshCenter[0], -_meshCenter[2], _meshCenter[1]); // z-up model
    rotate(pData->ModelViewMatrix, -90, 1, 0, 0); // z-up model
    inverseMatrix(pData->ModelViewInverseMatrix, pData->ModelViewMatrix);
    
    identity(pData->ProjectionMatrix);
    perspective(pData->ProjectionMatrix, 45.0, _cameraData.aspectRatio, 0.01f, 500.0);
    multMatrix(pData->ModelViewProjectionMatrix, pData->ModelViewMatrix, pData->ProjectionMatrix);
}

-(void)_initializeBuffers {
    _frameConstantsBuffer.alloc(_context.device, 1, @"frame constants");
}

-(void)_initializeCamera {
    _cameraData.dollyDistance = 4;
    _cameraData.rotationY = 30;
    _cameraData.rotationX = 0;
    _cameraData.aspectRatio = 1;
}

-(void)_initializeModels {
    initShapes();
    _loadedModels = [[NSMutableArray alloc] initWithCapacity:g_defaultShapes.size()];
    int i = 0;
    for(auto& shape : g_defaultShapes)
    {
        _loadedModels[i++] = [NSString stringWithUTF8String:shape.name.c_str()];
    }
    _currentModel = _loadedModels[0];
}


//Setters for triggering _needsRebuild on property change

-(void)setEndCapMode:(EndCap)endCapMode {
    _needsRebuild |= endCapMode != _endCapMode;
    _endCapMode = endCapMode;
}

-(void)setUseStageIn:(bool)useStageIn {
    _needsRebuild |= useStageIn != _useStageIn;
    _useStageIn = useStageIn;
}

-(void)setShadingMode:(ShadingMode)shadingMode {
    _needsRebuild |= shadingMode != _shadingMode;
    _shadingMode = shadingMode;
}

-(void)setKernelType:(KernelType)kernelType {
    _needsRebuild |= kernelType != _kernelType;
    _kernelType = kernelType;
}


-(void)setFVarBoundary:(FVarBoundary)fVarBoundary {
    _needsRebuild |= (fVarBoundary != _fVarBoundary);
    _fVarBoundary = fVarBoundary;
}

-(void)setCurrentModel:(NSString *)currentModel {
    _needsRebuild |= ![currentModel isEqualToString:_currentModel];
    _currentModel = currentModel;
}

-(void)setRefinementLevel:(unsigned int)refinementLevel {
    _needsRebuild |= refinementLevel != _refinementLevel;
    _refinementLevel = refinementLevel;
}

-(void)setUseSingleCrease:(bool)useSingleCrease {
    _needsRebuild |= useSingleCrease != _useSingleCrease;
    _useSingleCrease = useSingleCrease;
}

-(void)setUseInfinitelySharpPatch:(bool)useInfinitelySharpPatch {
    _needsRebuild |= useInfinitelySharpPatch != _useInfinitelySharpPatch;
    _useInfinitelySharpPatch = useInfinitelySharpPatch;
}

-(void)setUseRandomStart:(bool)useRandomStart {
    _needsRebuild |= useRandomStart != _useRandomStart;
    _useRandomStart = useRandomStart;
}

-(void)setUseAnimateParticles:(bool)useAnimateParticles {
    _needsRebuild |= useAnimateParticles != _useAnimateParticles;
    _useAnimateParticles = useAnimateParticles;
}

@end
