//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_MTL_CONTROL_MESH_DISPLAY_H
#define OPENSUBDIV_EXAMPLES_MTL_CONTROL_MESH_DISPLAY_H

#include <Metal/Metal.h>
#include <opensubdiv/far/topologyLevel.h>

class MTLControlMeshDisplay {
public:
    MTLControlMeshDisplay(id<MTLDevice> device, MTLRenderPipelineDescriptor* pipelineDescriptor);
    ~MTLControlMeshDisplay() = default;

    void Draw(id<MTLRenderCommandEncoder> encoder, 
    		  id<MTLBuffer> vertexBuffer,
              const float *modelViewProjectionMatrix);

    void SetTopology(OpenSubdiv::Far::TopologyLevel const &level);

    bool GetEdgesDisplay() const { return _displayEdges; }
    void SetEdgesDisplay(bool display) { _displayEdges = display; }
    bool GetVerticesDisplay() const { return _displayVertices; }
    void SetVerticesDisplay(bool display) { _displayVertices = display; }

private:
    bool createProgram(MTLRenderPipelineDescriptor* pipelineDescriptor);

    id<MTLDevice> _device;

    bool _displayEdges;
    bool _displayVertices;

    int _numEdges, _numPoints;
    id<MTLRenderPipelineState> _renderPipelineState;
    id<MTLBuffer> _vertexSharpnessBuffer;
    id<MTLBuffer> _edgeSharpnessBuffer;
    id<MTLBuffer> _edgeIndicesBuffer;
};

#endif  // OPENSUBDIV_EXAMPLES_MTL_CONTROL_MESH_DISPLAY_H
