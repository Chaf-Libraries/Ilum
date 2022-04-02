#include "ShaderCompiler.hpp"

#include "File/FileSystem.hpp"

#include <atlbase.h>
#include <dxcapi.h>

__pragma(warning(push, 0))
#include <shaderc/glslc/src/file_includer.h>
#include <shaderc/libshaderc_util/include/libshaderc_util/file_finder.h>
#include <shaderc/shaderc.hpp>
    __pragma(warning(pop))

        namespace Ilum
{
	// Shaderc
	static shaderc_compile_options_t            s_shaderc_opts;
	static shaderc_compiler_t                   s_shaderc_compiler;
	static shaderc_util::FileFinder             s_fileFinder;
	static std::unique_ptr<glslc::FileIncluder> s_includer;

	// DXC
	static CComPtr<IDxcUtils>     s_dxc_utils    = nullptr;
	static CComPtr<IDxcCompiler3> s_dxc_compiler = nullptr;
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
				return "lib_6_7";
			case VK_SHADER_STAGE_TASK_BIT_NV:
				return "ms_6_7";
			case VK_SHADER_STAGE_MESH_BIT_NV:
				return "as_6_7";
			default:
				return "";
		}

		return "";
	}

	inline shaderc_shader_kind get_shader_language(VkShaderStageFlags stage)
	{
		switch (stage)
		{
			case VK_SHADER_STAGE_VERTEX_BIT:
				return shaderc_vertex_shader;
			case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
				return shaderc_tess_control_shader;
			case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
				return shaderc_tess_evaluation_shader;
			case VK_SHADER_STAGE_GEOMETRY_BIT:
				return shaderc_geometry_shader;
			case VK_SHADER_STAGE_FRAGMENT_BIT:
				return shaderc_fragment_shader;
			case VK_SHADER_STAGE_COMPUTE_BIT:
				return shaderc_compute_shader;
			case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
				return shaderc_raygen_shader;
			case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
				return shaderc_anyhit_shader;
			case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
				return shaderc_closesthit_shader;
			case VK_SHADER_STAGE_MISS_BIT_KHR:
				return shaderc_miss_shader;
			case VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
				return shaderc_intersection_shader;
			case VK_SHADER_STAGE_CALLABLE_BIT_KHR:
				return shaderc_callable_shader;
			case VK_SHADER_STAGE_MESH_BIT_NV:
				return shaderc_task_shader;
			case VK_SHADER_STAGE_TASK_BIT_NV:
				return shaderc_mesh_shader;
			default:
				return shaderc_spirv_assembly;
		}
	}

	inline void init_shaderc_compiler()
	{
		s_shaderc_opts     = shaderc_compile_options_initialize();
		s_shaderc_compiler = shaderc_compiler_initialize();

		s_includer = std::make_unique<glslc::FileIncluder>(&s_fileFinder);

		shaderc_compile_options_set_include_callbacks(
		    s_shaderc_opts,
		    [](void *user_data, const char *requested_source, int type,
		       const char *requesting_source, size_t include_depth) {
			    auto *sub_includer = static_cast<shaderc::CompileOptions::IncluderInterface *>(user_data);
			    return sub_includer->GetInclude(
			        requested_source, static_cast<shaderc_include_type>(type),
			        requesting_source, include_depth);
		    },
		    [](void *user_data, shaderc_include_result *include_result) {
			    auto *sub_includer = static_cast<shaderc::CompileOptions::IncluderInterface *>(user_data);
			    return sub_includer->ReleaseInclude(include_result);
		    },
		    s_includer.get());

		shaderc_compile_options_set_generate_debug_info(s_shaderc_opts);
		shaderc_compile_options_set_optimization_level(s_shaderc_opts, shaderc_optimization_level_zero);
		shaderc_compile_options_set_target_env(s_shaderc_opts, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
		shaderc_compile_options_set_target_spirv(s_shaderc_opts, shaderc_spirv_version_1_5);
		shaderc_compile_options_set_auto_combined_image_sampler(s_shaderc_opts, true);
		shaderc_compile_options_set_auto_bind_uniforms(s_shaderc_opts, true);
	}

	inline bool error_check(IDxcResult * dxc_result)
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

	inline void destroy_shaderc_compiler()
	{
		shaderc_compile_options_release(s_shaderc_opts);
		shaderc_compiler_release(s_shaderc_compiler);
	}

	inline void init_dxc_compiler()
	{
		DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&s_dxc_utils));
		DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&s_dxc_compiler));
		s_dxc_utils->CreateDefaultIncludeHandler(&s_pDefaultIncludeHandler);
	}

	inline std::vector<uint32_t> shaderc_compile(const std::string &filename, const std::vector<uint8_t> &data, VkShaderStageFlags stage, Shader::Type type, const std::string &entry_point)
	{
		std::string source;
		source.resize(data.size());
		std::memcpy(source.data(), data.data(), data.size());
		source = std::string(source.c_str());

		if (type == Shader::Type::GLSL)
		{
			shaderc_compile_options_set_source_language(s_shaderc_opts, shaderc_source_language_glsl);
		}
		else if (type == Shader::Type::HLSL)
		{
			shaderc_compile_options_set_source_language(s_shaderc_opts, shaderc_source_language_hlsl);
		}

		shaderc_compilation_result_t res    = shaderc_compile_into_spv(s_shaderc_compiler, source.c_str(), source.size(), get_shader_language(stage), filename.c_str(), entry_point.c_str(), s_shaderc_opts);
		shaderc_compilation_status   status = shaderc_result_get_compilation_status(res);

		if (status != shaderc_compilation_status_success)
		{
			LOG_ERROR("Couldn't compile shader with built-in shaderc: {}", shaderc_result_get_error_message(res));

			if (res)
			{
				shaderc_result_release(res);
			}

			ASSERT(false);
			return {};
		}

		size_t sz = shaderc_result_get_length(res);

		std::vector<uint32_t> spirv(sz / 4);
		std::memcpy(spirv.data(), data.data(), data.size());

		memcpy(spirv.data(), shaderc_result_get_bytes(res), sz);

		shaderc_result_release(res);

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
		//arguments.emplace_back(L"-fspv-reflect");
		arguments.emplace_back(L"-fspv-target-env=vulkan1.2");
		arguments.emplace_back(L"-fspv-extension=SPV_KHR_ray_tracing");
		arguments.emplace_back(L"-fspv-extension=SPV_KHR_shader_draw_parameters");
		arguments.emplace_back(L"-fspv-extension=SPV_EXT_descriptor_indexing");

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
		init_shaderc_compiler();
		init_dxc_compiler();
	}

	void ShaderCompiler::destroy()
	{
		destroy_shaderc_compiler();
	}

	std::vector<uint32_t> ShaderCompiler::compile(const std::string &filename, const std::vector<uint8_t> &data, VkShaderStageFlags stage, Shader::Type type, const std::string &entry_point)
	{
		if (type == Shader::Type::HLSL)
		{
			return dxc_compile(filename, data, stage, type, entry_point);
		}
		else
		{
			return shaderc_compile(filename, data, stage, type, entry_point);
		}
	}
}        // namespace Ilum