//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "glLoader.h"

#include "../osd/glComputeEvaluator.h"
#include "../osd/glslPatchShaderSource.h"

#include "../far/error.h"
#include "../far/stencilTable.h"

#include <cassert>
#include <sstream>
#include <cstring>
#include <vector>


namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

static const char *shaderSource =
#include "../osd/glslComputeKernel.gen.h"
;

template <class T> GLuint
createSSBO(std::vector<T> const & src) {
    if (src.empty()) {
        return 0;
    }

    GLuint devicePtr = 0;

#if defined(GL_ARB_direct_state_access)
    if (OSD_OPENGL_HAS(ARB_direct_state_access)) {
        glCreateBuffers(1, &devicePtr);
        glNamedBufferData(devicePtr, src.size()*sizeof(T),
                          &src.at(0), GL_STATIC_DRAW);
    } else
#endif
    {
        GLint prev = 0;
        glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &prev);
        glGenBuffers(1, &devicePtr);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, devicePtr);
        glBufferData(GL_SHADER_STORAGE_BUFFER, src.size()*sizeof(T),
                     &src.at(0), GL_STATIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, prev);
    }

    return devicePtr;
}

GLStencilTableSSBO::GLStencilTableSSBO(
    Far::StencilTable const *stencilTable) {
    _numStencils = stencilTable->GetNumStencils();
    if (_numStencils > 0) {
        _sizes   = createSSBO(stencilTable->GetSizes());
        _offsets = createSSBO(stencilTable->GetOffsets());
        _indices = createSSBO(stencilTable->GetControlIndices());
        _weights = createSSBO(stencilTable->GetWeights());
        _duWeights = _dvWeights = 0;
        _duuWeights = _duvWeights = _dvvWeights = 0;
    } else {
        _sizes = _offsets = _indices = _weights = 0;
        _duWeights = _dvWeights = 0;
        _duuWeights = _duvWeights = _dvvWeights = 0;
    }
}

GLStencilTableSSBO::GLStencilTableSSBO(
    Far::LimitStencilTable const *limitStencilTable) {
    _numStencils = limitStencilTable->GetNumStencils();
    if (_numStencils > 0) {
        _sizes   = createSSBO(limitStencilTable->GetSizes());
        _offsets = createSSBO(limitStencilTable->GetOffsets());
        _indices = createSSBO(limitStencilTable->GetControlIndices());
        _weights = createSSBO(limitStencilTable->GetWeights());
        _duWeights = createSSBO(limitStencilTable->GetDuWeights());
        _dvWeights = createSSBO(limitStencilTable->GetDvWeights());
        _duuWeights = createSSBO(limitStencilTable->GetDuuWeights());
        _duvWeights = createSSBO(limitStencilTable->GetDuvWeights());
        _dvvWeights = createSSBO(limitStencilTable->GetDvvWeights());
    } else {
        _sizes = _offsets = _indices = _weights = 0;
        _duWeights = _dvWeights = 0;
        _duuWeights = _duvWeights = _dvvWeights = 0;
    }
}

GLStencilTableSSBO::~GLStencilTableSSBO() {
    if (_sizes)   glDeleteBuffers(1, &_sizes);
    if (_offsets) glDeleteBuffers(1, &_offsets);
    if (_indices) glDeleteBuffers(1, &_indices);
    if (_weights) glDeleteBuffers(1, &_weights);
    if (_duWeights) glDeleteBuffers(1, &_duWeights);
    if (_dvWeights) glDeleteBuffers(1, &_dvWeights);
    if (_duuWeights) glDeleteBuffers(1, &_duuWeights);
    if (_duvWeights) glDeleteBuffers(1, &_duvWeights);
    if (_dvvWeights) glDeleteBuffers(1, &_dvvWeights);
}

// ---------------------------------------------------------------------------


GLComputeEvaluator::GLComputeEvaluator()
    : _workGroupSize(64),
      _patchArraysSSBO(0) {
    std::memset((void*) &_stencilKernel, 0, sizeof(_stencilKernel));
    std::memset((void*) &_patchKernel, 0, sizeof(_patchKernel));

    // Initialize internal OpenGL loader library if necessary
    OpenSubdiv::internal::GLLoader::libraryInitializeGL();
}

GLComputeEvaluator::~GLComputeEvaluator() {
    if (_patchArraysSSBO) {
        glDeleteBuffers(1, &_patchArraysSSBO);
    }
}

static GLuint
compileKernel(BufferDescriptor const &srcDesc,
              BufferDescriptor const &dstDesc,
              BufferDescriptor const & duDesc,
              BufferDescriptor const & dvDesc,
              BufferDescriptor const & duuDesc,
              BufferDescriptor const & duvDesc,
              BufferDescriptor const & dvvDesc,
              const char *kernelDefine,
              int workGroupSize) {
    GLuint program = glCreateProgram();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);

    std::string patchBasisShaderSource =
        GLSLPatchShaderSource::GetPatchBasisShaderSource();
    const char *patchBasisShaderSourceDefine = "#define OSD_PATCH_BASIS_GLSL\n";

    std::ostringstream defines;
    defines << "#define LENGTH "     << srcDesc.length << "\n"
            << "#define SRC_STRIDE " << srcDesc.stride << "\n"
            << "#define DST_STRIDE " << dstDesc.stride << "\n"
            << "#define WORK_GROUP_SIZE " << workGroupSize << "\n"
            << kernelDefine << "\n"
            << patchBasisShaderSourceDefine << "\n";

    bool deriv1 = (duDesc.length > 0 || dvDesc.length > 0);
    bool deriv2 = (duuDesc.length > 0 || duvDesc.length > 0 || dvvDesc.length > 0);
    if (deriv1) {
        defines << "#define OPENSUBDIV_GLSL_COMPUTE_USE_1ST_DERIVATIVES\n";
    }
    if (deriv2) {
        defines << "#define OPENSUBDIV_GLSL_COMPUTE_USE_2ND_DERIVATIVES\n";
    }

    std::string defineStr = defines.str();

    const char *shaderSources[4] = {"#version 430\n", 0, 0, 0};

    shaderSources[1] = defineStr.c_str();
    shaderSources[2] = patchBasisShaderSource.c_str();
    shaderSources[3] = shaderSource;
    glShaderSource(shader, 4, shaderSources, NULL);
    glCompileShader(shader);
    glAttachShader(program, shader);

    GLint linked = 0;
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &linked);

    if (linked == GL_FALSE) {
        char buffer[1024];
        glGetShaderInfoLog(shader, 1024, NULL, buffer);
        Far::Error(Far::FAR_RUNTIME_ERROR, buffer);

        glGetProgramInfoLog(program, 1024, NULL, buffer);
        Far::Error(Far::FAR_RUNTIME_ERROR, buffer);

        glDeleteProgram(program);
        return 0;
    }

    glDeleteShader(shader);

    return program;
}

bool
GLComputeEvaluator::Compile(BufferDescriptor const &srcDesc,
                            BufferDescriptor const &dstDesc,
                            BufferDescriptor const &duDesc,
                            BufferDescriptor const &dvDesc,
                            BufferDescriptor const &duuDesc,
                            BufferDescriptor const &duvDesc,
                            BufferDescriptor const &dvvDesc) {

    // create a stencil kernel
    if (!_stencilKernel.Compile(srcDesc, dstDesc,
                                duDesc, dvDesc,
                                duuDesc, duvDesc, dvvDesc,
                                _workGroupSize)) {
        return false;
    }

    // create a patch kernel
    if (!_patchKernel.Compile(srcDesc, dstDesc,
                              duDesc, dvDesc,
                              duuDesc, duvDesc, dvvDesc,
                              _workGroupSize)) {
        return false;
    }

    // create a patch arrays buffer
    if (!_patchArraysSSBO) {
        glGenBuffers(1, &_patchArraysSSBO);
    }

    return true;
}

/* static */
void
GLComputeEvaluator::Synchronize(void * /*kernel*/) {
    // XXX: this is currently just for the performance measuring purpose.
    // need to be reimplemented by fence and sync.
    glFinish();
}

bool
GLComputeEvaluator::EvalStencils(
    ID srcBuffer, BufferDescriptor const &srcDesc,
    ID dstBuffer, BufferDescriptor const &dstDesc,
    ID duBuffer,  BufferDescriptor const &duDesc,
    ID dvBuffer,  BufferDescriptor const &dvDesc,
    ID sizesBuffer,
    ID offsetsBuffer,
    ID indicesBuffer,
    ID weightsBuffer,
    ID duWeightsBuffer,
    ID dvWeightsBuffer,
    int start, int end) const {

    return EvalStencils(srcBuffer, srcDesc,
                        dstBuffer, dstDesc,
                        duBuffer, duDesc,
                        dvBuffer, dvDesc,
                        0, BufferDescriptor(),
                        0, BufferDescriptor(),
                        0, BufferDescriptor(),
                        sizesBuffer, offsetsBuffer, indicesBuffer,
                        weightsBuffer,
                        duWeightsBuffer, dvWeightsBuffer,
                        0, 0, 0,
                        start, end);
}

bool
GLComputeEvaluator::EvalStencils(
    ID srcBuffer, BufferDescriptor const &srcDesc,
    ID dstBuffer, BufferDescriptor const &dstDesc,
    ID duBuffer,  BufferDescriptor const &duDesc,
    ID dvBuffer,  BufferDescriptor const &dvDesc,
    ID duuBuffer, BufferDescriptor const &duuDesc,
    ID duvBuffer, BufferDescriptor const &duvDesc,
    ID dvvBuffer, BufferDescriptor const &dvvDesc,
    ID sizesBuffer,
    ID offsetsBuffer,
    ID indicesBuffer,
    ID weightsBuffer,
    ID duWeightsBuffer,
    ID dvWeightsBuffer,
    ID duuWeightsBuffer,
    ID duvWeightsBuffer,
    ID dvvWeightsBuffer,
    int start, int end) const {

    if (!_stencilKernel.program) return false;
    int count = end - start;
    if (count <= 0) {
        return true;
    }

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, srcBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, dstBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, duBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, dvBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, duuBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, duvBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, dvvBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, sizesBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, offsetsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, indicesBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, weightsBuffer);
    if (duWeightsBuffer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, duWeightsBuffer);
    if (dvWeightsBuffer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, dvWeightsBuffer);
    if (duuWeightsBuffer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, duuWeightsBuffer);
    if (duvWeightsBuffer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, duvWeightsBuffer);
    if (dvvWeightsBuffer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, dvvWeightsBuffer);

    GLint activeProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM, &activeProgram);
    glUseProgram(_stencilKernel.program);

    glUniform1i(_stencilKernel.uniformStart,     start);
    glUniform1i(_stencilKernel.uniformEnd,       end);
    glUniform1i(_stencilKernel.uniformSrcOffset, srcDesc.offset);
    glUniform1i(_stencilKernel.uniformDstOffset, dstDesc.offset);
    if (_stencilKernel.uniformDuDesc > 0) {
        glUniform3i(_stencilKernel.uniformDuDesc,
                    duDesc.offset, duDesc.length, duDesc.stride);
    }
    if (_stencilKernel.uniformDvDesc > 0) {
        glUniform3i(_stencilKernel.uniformDvDesc,
                    dvDesc.offset, dvDesc.length, dvDesc.stride);
    }
    if (_stencilKernel.uniformDuuDesc > 0) {
        glUniform3i(_stencilKernel.uniformDuuDesc,
                    duuDesc.offset, duuDesc.length, duuDesc.stride);
    }
    if (_stencilKernel.uniformDuvDesc > 0) {
        glUniform3i(_stencilKernel.uniformDuvDesc,
                    duvDesc.offset, duvDesc.length, duvDesc.stride);
    }
    if (_stencilKernel.uniformDvvDesc > 0) {
        glUniform3i(_stencilKernel.uniformDvvDesc,
                    dvvDesc.offset, dvvDesc.length, dvvDesc.stride);
    }

    glDispatchCompute((count + _workGroupSize - 1) / _workGroupSize, 1, 1);

    glUseProgram(activeProgram);

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    for (int i = 0; i < 16; ++i) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
    }

    return true;
}

bool
GLComputeEvaluator::EvalPatches(
    ID srcBuffer, BufferDescriptor const &srcDesc,
    ID dstBuffer, BufferDescriptor const &dstDesc,
    ID duBuffer,  BufferDescriptor const &duDesc,
    ID dvBuffer,  BufferDescriptor const &dvDesc,
    int numPatchCoords,
    ID patchCoordsBuffer,
    const PatchArrayVector &patchArrays,
    ID patchIndexBuffer,
    ID patchParamsBuffer) const {

    return EvalPatches(srcBuffer, srcDesc,
                       dstBuffer, dstDesc,
                       duBuffer, duDesc,
                       dvBuffer, dvDesc,
                       0, BufferDescriptor(),
                       0, BufferDescriptor(),
                       0, BufferDescriptor(),
                       numPatchCoords,
                       patchCoordsBuffer,
                       patchArrays,
                       patchIndexBuffer,
                       patchParamsBuffer);
}

bool
GLComputeEvaluator::EvalPatches(
    ID srcBuffer, BufferDescriptor const &srcDesc,
    ID dstBuffer, BufferDescriptor const &dstDesc,
    ID duBuffer,  BufferDescriptor const &duDesc,
    ID dvBuffer,  BufferDescriptor const &dvDesc,
    ID duuBuffer, BufferDescriptor const &duuDesc,
    ID duvBuffer, BufferDescriptor const &duvDesc,
    ID dvvBuffer, BufferDescriptor const &dvvDesc,
    int numPatchCoords,
    ID patchCoordsBuffer,
    const PatchArrayVector &patchArrays,
    ID patchIndexBuffer,
    ID patchParamsBuffer) const {

    if (!_patchKernel.program) return false;

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, srcBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, dstBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, duBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, dvBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, duuBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, duvBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, dvvBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, patchCoordsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, patchIndexBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, patchParamsBuffer);

    GLint activeProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM, &activeProgram);
    glUseProgram(_patchKernel.program);

    glUniform1i(_patchKernel.uniformSrcOffset, srcDesc.offset);
    glUniform1i(_patchKernel.uniformDstOffset, dstDesc.offset);

    int patchArraySize = sizeof(PatchArray);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _patchArraysSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
        patchArrays.size()*patchArraySize, NULL, GL_STATIC_DRAW);
    for (int i=0; i<(int)patchArrays.size(); ++i) {
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
            i*patchArraySize, sizeof(PatchArray), &patchArrays[i]);
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _patchArraysSSBO);

    if (_patchKernel.uniformDuDesc > 0) {
        glUniform3i(_patchKernel.uniformDuDesc,
                    duDesc.offset, duDesc.length, duDesc.stride);
    }
    if (_patchKernel.uniformDvDesc > 0) {
        glUniform3i(_patchKernel.uniformDvDesc,
                    dvDesc.offset, dvDesc.length, dvDesc.stride);
    }
    if (_patchKernel.uniformDuuDesc > 0) {
        glUniform3i(_patchKernel.uniformDuuDesc,
                    duuDesc.offset, duuDesc.length, duuDesc.stride);
    }
    if (_patchKernel.uniformDuvDesc > 0) {
        glUniform3i(_patchKernel.uniformDuvDesc,
                    duvDesc.offset, duvDesc.length, duvDesc.stride);
    }
    if (_patchKernel.uniformDvvDesc > 0) {
        glUniform3i(_patchKernel.uniformDvvDesc,
                    dvvDesc.offset, dvvDesc.length, dvvDesc.stride);
    }

    glDispatchCompute((numPatchCoords + _workGroupSize - 1) / _workGroupSize, 1, 1);

    glUseProgram(activeProgram);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, 0);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, 0);

    return true;
}
// ---------------------------------------------------------------------------

GLComputeEvaluator::_StencilKernel::_StencilKernel() : program(0) {
}
GLComputeEvaluator::_StencilKernel::~_StencilKernel() {
    if (program) {
        glDeleteProgram(program);
    }
}

bool
GLComputeEvaluator::_StencilKernel::Compile(BufferDescriptor const &srcDesc,
                                            BufferDescriptor const &dstDesc,
                                            BufferDescriptor const &duDesc,
                                            BufferDescriptor const &dvDesc,
                                            BufferDescriptor const &duuDesc,
                                            BufferDescriptor const &duvDesc,
                                            BufferDescriptor const &dvvDesc,
                                            int workGroupSize) {
    // create stencil kernel
    if (program) {
        glDeleteProgram(program);
    }

    const char * kernelDefine =
        "#define OPENSUBDIV_GLSL_COMPUTE_KERNEL_EVAL_STENCILS\n";

    program = compileKernel(srcDesc, dstDesc,
                            duDesc, dvDesc, duuDesc, duvDesc, dvvDesc,
                            kernelDefine, workGroupSize);
    if (program == 0) return false;

    // cache uniform locations (TODO: use uniform block)
    uniformStart     = glGetUniformLocation(program, "batchStart");
    uniformEnd       = glGetUniformLocation(program, "batchEnd");
    uniformSrcOffset = glGetUniformLocation(program, "srcOffset");
    uniformDstOffset = glGetUniformLocation(program, "dstOffset");
    uniformDuDesc    = glGetUniformLocation(program, "duDesc");
    uniformDvDesc    = glGetUniformLocation(program, "dvDesc");
    uniformDuuDesc   = glGetUniformLocation(program, "duuDesc");
    uniformDuvDesc   = glGetUniformLocation(program, "duvDesc");
    uniformDvvDesc   = glGetUniformLocation(program, "dvvDesc");

    return true;
}

// ---------------------------------------------------------------------------

GLComputeEvaluator::_PatchKernel::_PatchKernel() : program(0) {
}
GLComputeEvaluator::_PatchKernel::~_PatchKernel() {
    if (program) {
        glDeleteProgram(program);
    }
}

bool
GLComputeEvaluator::_PatchKernel::Compile(BufferDescriptor const &srcDesc,
                                          BufferDescriptor const &dstDesc,
                                          BufferDescriptor const &duDesc,
                                          BufferDescriptor const &dvDesc,
                                          BufferDescriptor const &duuDesc,
                                          BufferDescriptor const &duvDesc,
                                          BufferDescriptor const &dvvDesc,
                                          int workGroupSize) {
    // create stencil kernel
    if (program) {
        glDeleteProgram(program);
    }

    const char * kernelDefine =
        "#define OPENSUBDIV_GLSL_COMPUTE_KERNEL_EVAL_PATCHES\n";

    program = compileKernel(srcDesc, dstDesc,
                            duDesc, dvDesc, duuDesc, duvDesc, dvvDesc,
                            kernelDefine, workGroupSize);
    if (program == 0) return false;

    // cache uniform locations
    uniformSrcOffset  = glGetUniformLocation(program, "srcOffset");
    uniformDstOffset  = glGetUniformLocation(program, "dstOffset");
    uniformPatchArray = glGetUniformLocation(program, "patchArray");
    uniformDuDesc     = glGetUniformLocation(program, "duDesc");
    uniformDvDesc     = glGetUniformLocation(program, "dvDesc");
    uniformDuuDesc    = glGetUniformLocation(program, "duuDesc");
    uniformDuvDesc    = glGetUniformLocation(program, "duvDesc");
    uniformDvvDesc    = glGetUniformLocation(program, "dvvDesc");

    return true;
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
