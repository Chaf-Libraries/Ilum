#include "ShaderCompiler.hpp"

#include <glslang/Include/ResourceLimits.h>
#include <SPIRV/GLSL.std.450.h>
#include <SPIRV/GlslangToSpv.h>

#include <spirv_glsl.hpp>

namespace glslang
{
const TBuiltInResource DefaultTBuiltInResource = {
    /* .MaxLights = */ 32,
    /* .MaxClipPlanes = */ 6,
    /* .MaxTextureUnits = */ 32,
    /* .MaxTextureCoords = */ 32,
    /* .MaxVertexAttribs = */ 64,
    /* .MaxVertexUniformComponents = */ 4096,
    /* .MaxVaryingFloats = */ 64,
    /* .MaxVertexTextureImageUnits = */ 32,
    /* .MaxCombinedTextureImageUnits = */ 80,
    /* .MaxTextureImageUnits = */ 32,
    /* .MaxFragmentUniformComponents = */ 4096,
    /* .MaxDrawBuffers = */ 32,
    /* .MaxVertexUniformVectors = */ 128,
    /* .MaxVaryingVectors = */ 8,
    /* .MaxFragmentUniformVectors = */ 16,
    /* .MaxVertexOutputVectors = */ 16,
    /* .MaxFragmentInputVectors = */ 15,
    /* .MinProgramTexelOffset = */ -8,
    /* .MaxProgramTexelOffset = */ 7,
    /* .MaxClipDistances = */ 8,
    /* .MaxComputeWorkGroupCountX = */ 65535,
    /* .MaxComputeWorkGroupCountY = */ 65535,
    /* .MaxComputeWorkGroupCountZ = */ 65535,
    /* .MaxComputeWorkGroupSizeX = */ 1024,
    /* .MaxComputeWorkGroupSizeY = */ 1024,
    /* .MaxComputeWorkGroupSizeZ = */ 64,
    /* .MaxComputeUniformComponents = */ 1024,
    /* .MaxComputeTextureImageUnits = */ 16,
    /* .MaxComputeImageUniforms = */ 8,
    /* .MaxComputeAtomicCounters = */ 8,
    /* .MaxComputeAtomicCounterBuffers = */ 1,
    /* .MaxVaryingComponents = */ 60,
    /* .MaxVertexOutputComponents = */ 64,
    /* .MaxGeometryInputComponents = */ 64,
    /* .MaxGeometryOutputComponents = */ 128,
    /* .MaxFragmentInputComponents = */ 128,
    /* .MaxImageUnits = */ 8,
    /* .MaxCombinedImageUnitsAndFragmentOutputs = */ 8,
    /* .MaxCombinedShaderOutputResources = */ 8,
    /* .MaxImageSamples = */ 0,
    /* .MaxVertexImageUniforms = */ 0,
    /* .MaxTessControlImageUniforms = */ 0,
    /* .MaxTessEvaluationImageUniforms = */ 0,
    /* .MaxGeometryImageUniforms = */ 0,
    /* .MaxFragmentImageUniforms = */ 8,
    /* .MaxCombinedImageUniforms = */ 8,
    /* .MaxGeometryTextureImageUnits = */ 16,
    /* .MaxGeometryOutputVertices = */ 256,
    /* .MaxGeometryTotalOutputComponents = */ 1024,
    /* .MaxGeometryUniformComponents = */ 1024,
    /* .MaxGeometryVaryingComponents = */ 64,
    /* .MaxTessControlInputComponents = */ 128,
    /* .MaxTessControlOutputComponents = */ 128,
    /* .MaxTessControlTextureImageUnits = */ 16,
    /* .MaxTessControlUniformComponents = */ 1024,
    /* .MaxTessControlTotalOutputComponents = */ 4096,
    /* .MaxTessEvaluationInputComponents = */ 128,
    /* .MaxTessEvaluationOutputComponents = */ 128,
    /* .MaxTessEvaluationTextureImageUnits = */ 16,
    /* .MaxTessEvaluationUniformComponents = */ 1024,
    /* .MaxTessPatchComponents = */ 120,
    /* .MaxPatchVertices = */ 32,
    /* .MaxTessGenLevel = */ 64,
    /* .MaxViewports = */ 16,
    /* .MaxVertexAtomicCounters = */ 0,
    /* .MaxTessControlAtomicCounters = */ 0,
    /* .MaxTessEvaluationAtomicCounters = */ 0,
    /* .MaxGeometryAtomicCounters = */ 0,
    /* .MaxFragmentAtomicCounters = */ 8,
    /* .MaxCombinedAtomicCounters = */ 8,
    /* .MaxAtomicCounterBindings = */ 1,
    /* .MaxVertexAtomicCounterBuffers = */ 0,
    /* .MaxTessControlAtomicCounterBuffers = */ 0,
    /* .MaxTessEvaluationAtomicCounterBuffers = */ 0,
    /* .MaxGeometryAtomicCounterBuffers = */ 0,
    /* .MaxFragmentAtomicCounterBuffers = */ 1,
    /* .MaxCombinedAtomicCounterBuffers = */ 1,
    /* .MaxAtomicCounterBufferSize = */ 16384,
    /* .MaxTransformFeedbackBuffers = */ 4,
    /* .MaxTransformFeedbackInterleavedComponents = */ 64,
    /* .MaxCullDistances = */ 8,
    /* .MaxCombinedClipAndCullDistances = */ 8,
    /* .MaxSamples = */ 4,
    /* .maxMeshOutputVerticesNV = */ 256,
    /* .maxMeshOutputPrimitivesNV = */ 512,
    /* .maxMeshWorkGroupSizeX_NV = */ 32,
    /* .maxMeshWorkGroupSizeY_NV = */ 1,
    /* .maxMeshWorkGroupSizeZ_NV = */ 1,
    /* .maxTaskWorkGroupSizeX_NV = */ 32,
    /* .maxTaskWorkGroupSizeY_NV = */ 1,
    /* .maxTaskWorkGroupSizeZ_NV = */ 1,
    /* .maxMeshViewCountNV = */ 4,
    /* .maxDualSourceDrawBuffersEXT = */ 1,

    /* .limits = */ {
        /* .nonInductiveForLoops = */ 1,
        /* .whileLoops = */ 1,
        /* .doWhileLoops = */ 1,
        /* .generalUniformIndexing = */ 1,
        /* .generalAttributeMatrixVectorIndexing = */ 1,
        /* .generalVaryingIndexing = */ 1,
        /* .generalSamplerIndexing = */ 1,
        /* .generalVariableIndexing = */ 1,
        /* .generalConstantMatrixVectorIndexing = */ 1,
    }};
}

namespace Ilum
{
inline EShLanguage get_shader_language(VkShaderStageFlags stage)
{
	switch (stage)
	{
		case VK_SHADER_STAGE_VERTEX_BIT:
			return EShLangVertex;
		case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
			return EShLangTessControl;
		case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
			return EShLangTessEvaluation;
		case VK_SHADER_STAGE_GEOMETRY_BIT:
			return EShLangGeometry;
		case VK_SHADER_STAGE_FRAGMENT_BIT:
			return EShLangFragment;
		case VK_SHADER_STAGE_COMPUTE_BIT:
			return EShLangCompute;
		case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
			return EShLangRayGen;
		case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
			return EShLangAnyHit;
		case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
			return EShLangClosestHit;
		case VK_SHADER_STAGE_MISS_BIT_KHR:
			return EShLangMiss;
		case VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
			return EShLangIntersect;
		case VK_SHADER_STAGE_CALLABLE_BIT_KHR:
			return EShLangCallable;
		case VK_SHADER_STAGE_MESH_BIT_NV:
			return EShLangMeshNV;
		case VK_SHADER_STAGE_TASK_BIT_NV:
			return EShLangTaskNV;
		default:
			return EShLangCount;
	}
}

inline std::vector<uint32_t> compileGLSL(const std::vector<uint8_t> &data, VkShaderStageFlags stage)
{
	glslang::InitializeProcess();

	EShMessages msgs = static_cast<EShMessages>(EShMsgDefault | EShMsgVulkanRules | EShMsgSpvRules);

	EShLanguage lang = get_shader_language(stage);

	std::string info_log = "";

	const char *file_name_list[1] = {""};
	const char *shader_source     = reinterpret_cast<const char *>(data.data());

	glslang::TShader shader(lang);
	shader.setStringsWithLengthsAndNames(&shader_source, nullptr, file_name_list, 1);
	shader.setEntryPoint("main");
	shader.setSourceEntryPoint("main");
	shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_4);

	if (!shader.parse(&glslang::DefaultTBuiltInResource, 100, false, msgs))
	{
		info_log = std::string(shader.getInfoLog()) + "\n" + std::string(shader.getInfoDebugLog());
		LOG_ERROR(info_log);
		return {};
	}

	// Add shader to a new program
	glslang::TProgram program;
	program.addShader(&shader);

	// Link program
	if (!program.link(msgs))
	{
		info_log = std::string(program.getInfoLog()) + "\n" + std::string(program.getInfoDebugLog());
		LOG_ERROR(info_log);
		return {};
	}

	// Save info log
	if (shader.getInfoLog())
	{
		info_log = std::string(shader.getInfoLog()) + "\n" + std::string(shader.getInfoDebugLog());
	}

	if (program.getInfoLog())
	{
		info_log = std::string(program.getInfoLog()) + "\n" + std::string(program.getInfoDebugLog());
	}

	glslang::TIntermediate *intermediate = program.getIntermediate(lang);

	if (!intermediate)
	{
		info_log += "Failed to get shared intermediate code!\n";
		LOG_ERROR(info_log);
		return {};
	}

	spv::SpvBuildLogger logger;

	std::vector<uint32_t> spirv;
	std::vector<uint8_t>  result;

	glslang::GlslangToSpv(*intermediate, spirv, &logger);
	result.resize(spirv.size() * 4);
	std::memcpy(result.data(), spirv.data(), result.size());

	info_log += logger.getAllMessages() + "\n";

	glslang::FinalizeProcess();
	return spirv;
}

inline std::vector<uint32_t> compileHLSL(const std::vector<uint8_t> &data, VkShaderStageFlags stage)
{
    // TODO: Use DXC
	return {};
}

std::vector<uint32_t> ShaderCompiler::compile(const std::vector<uint8_t> &data, VkShaderStageFlags stage, Shader::Type type)
{
    if (type == Shader::Type::GLSL)
    {
		return compileGLSL(data, stage);
    }
    else if (type == Shader::Type::HLSL)
    {
		return compileHLSL(data, stage);
    }
    // SPIR-V
	std::vector<uint32_t> spirv(data.size() / 4);
	std::memcpy(spirv.data(), data.data(), data.size());

	return spirv;
}
}        // namespace Ilum