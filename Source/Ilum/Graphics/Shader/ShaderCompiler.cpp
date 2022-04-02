#include "ShaderCompiler.hpp"

__pragma(warning(push, 0))
#include <shaderc/shaderc.hpp>
    __pragma(warning(pop))

        namespace Ilum
{
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

	std::vector<uint32_t> ShaderCompiler::compile(const std::string &filename, const std::vector<uint8_t> &data, VkShaderStageFlags stage, Shader::Type type, const std::string &entry_point)
	{
		shaderc_compile_options_t opts = shaderc_compile_options_initialize();

		shaderc_compiler_t compiler = shaderc_compiler_initialize();

		if (type == Shader::Type::GLSL)
		{
			shaderc_compile_options_set_source_language(opts, shaderc_source_language_glsl);
		}
		else if (type == Shader::Type::HLSL)
		{
			shaderc_compile_options_set_source_language(opts, shaderc_source_language_hlsl);
		}

		shaderc_compile_options_set_generate_debug_info(opts);
		shaderc_compile_options_set_optimization_level(opts, shaderc_optimization_level_zero);
		shaderc_compile_options_set_target_env(opts, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
		shaderc_compile_options_set_target_spirv(opts, shaderc_spirv_version_1_5);

		std::string source;
		source.resize(data.size());
		std::memcpy(source.data(), data.data(), data.size());
		source = std::string(source.c_str());

		shaderc_compilation_result_t res = shaderc_compile_into_spv(compiler, source.c_str(), source.size(), get_shader_language(stage), filename.c_str(), entry_point.c_str(), opts);

		shaderc_compilation_status status = shaderc_result_get_compilation_status(res);

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

		shaderc_compile_options_release(opts);

		return spirv;
	}
}        // namespace Ilum