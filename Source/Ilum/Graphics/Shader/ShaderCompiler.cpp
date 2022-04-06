#include "ShaderCompiler.hpp"

#include "File/FileSystem.hpp"

#include <atlbase.h>
#include <dxcapi.h>

#include <glslang/SPIRV/GLSL.std.450.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <glslang/Include/ResourceLimits.h>

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
// DXC
static CComPtr<IDxcUtils>          s_dxc_utils              = nullptr;
static CComPtr<IDxcCompiler3>      s_dxc_compiler           = nullptr;
static CComPtr<IDxcIncludeHandler> s_pDefaultIncludeHandler = nullptr;

std::wstring to_wstring(const std::string &str)
{
	const auto slength = static_cast<int>(str.length()) + 1;
	const auto len     = MultiByteToWideChar(CP_ACP, 0, str.c_str(), slength, nullptr, 0);
	const auto buf     = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, str.c_str(), slength, buf, len);
	std::wstring result(buf);
	delete[] buf;
	return result;
}

inline std::string get_target_profile(VkShaderStageFlags stage)
{
	switch (stage)
	{
		case VK_SHADER_STAGE_VERTEX_BIT:
			return "vs_6_7";
		case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
			return "hs_6_7";
		case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
			return "ds_6_7";
		case VK_SHADER_STAGE_GEOMETRY_BIT:
			return "gs_6_7";
		case VK_SHADER_STAGE_FRAGMENT_BIT:
			return "ps_6_7";
		case VK_SHADER_STAGE_COMPUTE_BIT:
			return "cs_6_7";
		case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
		case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
		case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
		case VK_SHADER_STAGE_MISS_BIT_KHR:
		case VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
		case VK_SHADER_STAGE_CALLABLE_BIT_KHR:
			return "lib_6_6";
		case VK_SHADER_STAGE_TASK_BIT_NV:
			return "ms_6_7";
		case VK_SHADER_STAGE_MESH_BIT_NV:
			return "as_6_7";
		default:
			return "";
	}

	return "";
}

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

inline void init_glslang_compiler()
{
	glslang::InitializeProcess();
}

inline bool error_check(IDxcResult *dxc_result)
{
	// Get error buffer
	IDxcBlobEncoding *error_buffer = nullptr;
	HRESULT           result       = dxc_result->GetErrorBuffer(&error_buffer);
	if (SUCCEEDED(result))
	{
		// Log info, warnings and errors
		std::stringstream ss(std::string(static_cast<char *>(error_buffer->GetBufferPointer()), error_buffer->GetBufferSize()));
		std::string       line;
		while (getline(ss, line, '\n'))
		{
			if (line.find("error") != std::string::npos)
			{
				LOG_ERROR(line);
			}
			else if (line.find("warning") != std::string::npos)
			{
				LOG_WARN(line);
			}
			else if (!line.empty())
			{
				LOG_INFO(line);
			}
		}
	}
	else
	{
		LOG_ERROR("Failed to get error buffer");
	}

	// Release error buffer
	if (error_buffer)
	{
		error_buffer->Release();
		error_buffer = nullptr;
	}

	// Return status
	dxc_result->GetStatus(&result);
	return result == S_OK;
}

inline void destroy_glslang_compiler()
{
	glslang::FinalizeProcess();
}

inline void init_dxc_compiler()
{
	DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&s_dxc_utils));
	DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&s_dxc_compiler));
	s_dxc_utils->CreateDefaultIncludeHandler(&s_pDefaultIncludeHandler);
}

inline std::vector<uint32_t> glslang_compile(const std::string &filename, const std::vector<uint8_t> &data, VkShaderStageFlags stage, Shader::Type type, const std::string &entry_point)
{
	EShMessages msgs = static_cast<EShMessages>(EShMsgDefault | EShMsgVulkanRules | EShMsgSpvRules);

	EShLanguage lang = get_shader_language(stage);

	std::string info_log = "";

	const char *file_name_list[1] = {""};
	const char *shader_source     = reinterpret_cast<const char *>(data.data());

	glslang::TShader shader(lang);
	shader.setStringsWithLengthsAndNames(&shader_source, nullptr, file_name_list, 1);
	shader.setEntryPoint("main");
	shader.setSourceEntryPoint("main");
	shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);
	shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);

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

	return spirv;
}

inline std::vector<uint32_t> dxc_compile(const std::string &filename, const std::vector<uint8_t> &data, VkShaderStageFlags stage, Shader::Type type, const std::string &entry_point)
{
	std::string source;
	source.resize(data.size());
	std::memcpy(source.data(), data.data(), data.size());
	source = std::string(source.c_str());

	DxcBuffer                 dxc_buffer;
	CComPtr<IDxcBlobEncoding> blob_encoding = nullptr;
	{
		if (FAILED(s_dxc_utils->CreateBlobFromPinned(source.c_str(), static_cast<uint32_t>(source.size()), CP_UTF8, &blob_encoding)))
		{
			LOG_ERROR("Failed to load shader source.");
			return {};
		}

		dxc_buffer.Ptr      = blob_encoding->GetBufferPointer();
		dxc_buffer.Size     = blob_encoding->GetBufferSize();
		dxc_buffer.Encoding = DXC_CP_ACP;        // Assume BOM says UTF8 or UTF16 or this is ANSI text.
	}

	std::vector<std::wstring> arguments;

	// Compile arguments
	arguments.emplace_back(L"-E");
	arguments.emplace_back(to_wstring(entry_point));
	arguments.emplace_back(to_wstring(filename));
	arguments.emplace_back(L"-T");
	arguments.emplace_back(to_wstring(get_target_profile(stage)));
	arguments.emplace_back(L"-I");
	arguments.emplace_back(to_wstring(FileSystem::getFileDirectory(filename)));
	arguments.emplace_back(L"-H");
	arguments.emplace_back(L"-spirv");
	arguments.emplace_back(L"-fspv-target-env=vulkan1.2");
	arguments.emplace_back(L"-fspv-extension=SPV_KHR_ray_tracing");
	arguments.emplace_back(L"-fspv-extension=SPV_KHR_shader_draw_parameters");
	arguments.emplace_back(L"-fspv-extension=SPV_EXT_descriptor_indexing");
	arguments.emplace_back(L"-fspv-extension=SPV_EXT_shader_viewport_index_layer");

#ifdef _DEBUG
	arguments.emplace_back(L"-Od");        // Disable optimizations
	arguments.emplace_back(L"-Zi");        // Enable debug information
#endif

	// Convert arguments to LPCWSTR
	std::vector<LPCWSTR> arguments_lpcwstr;
	arguments_lpcwstr.reserve(arguments.size());
	for (const std::wstring &wstr : arguments)
	{
		arguments_lpcwstr.emplace_back(wstr.c_str());
	}

	// Compile
	IDxcResult *dxc_result = nullptr;
	s_dxc_compiler->Compile(
	    &dxc_buffer,                                            // Source text to compile
	    arguments_lpcwstr.data(),                               // Array of pointers to arguments
	    static_cast<uint32_t>(arguments_lpcwstr.size()),        // Number of arguments
	    s_pDefaultIncludeHandler,                               // don't use an include handler
	    IID_PPV_ARGS(&dxc_result)                               // IDxcResult: status, buffer, and errors
	);

	// Check for errors
	if (!error_check(dxc_result))
	{
		LOG_ERROR("Failed to compile");

		if (dxc_result)
		{
			dxc_result->Release();
			dxc_result = nullptr;
		}
		else
		{
			ASSERT(false);
		}
	}

	IDxcBlob *shader_buffer = nullptr;
	dxc_result->GetResult(&shader_buffer);

	std::vector<uint32_t> spirv(shader_buffer->GetBufferSize() / 4);
	std::memcpy(spirv.data(), shader_buffer->GetBufferPointer(), shader_buffer->GetBufferSize());

	return spirv;
}

void ShaderCompiler::init()
{
	init_glslang_compiler();
	init_dxc_compiler();
}

void ShaderCompiler::destroy()
{
	destroy_glslang_compiler();
}

std::vector<uint32_t> ShaderCompiler::compile(const std::string &filename, const std::vector<uint8_t> &data, VkShaderStageFlags stage, Shader::Type type, const std::string &entry_point)
{
	if (type == Shader::Type::HLSL)
	{
		return dxc_compile(filename, data, stage, type, entry_point);
	}
	else
	{
		return glslang_compile(filename, data, stage, type, entry_point);
	}
}
}        // namespace Ilum