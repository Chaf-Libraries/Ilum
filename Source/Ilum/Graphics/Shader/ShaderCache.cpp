#include "ShaderCache.hpp"
#include "ShaderCompiler.hpp"

#include "File/FileSystem.hpp"

#include "Device/LogicalDevice.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Vulkan/Vulkan.hpp"

namespace Ilum
{
inline std::vector<std::string> split(const std::string &input, char delim)
{
	std::vector<std::string> tokens;

	std::stringstream sstream(input);
	std::string       token;
	while (std::getline(sstream, token, delim))
	{
		tokens.push_back(token);
	}

	return tokens;
}

inline std::vector<std::string> precompile_shader(const std::string &source, const std::string &include_dir)
{
	std::vector<std::string> final_file;

	auto lines = split(source, '\n');

	for (auto &line : lines)
	{
		if (line.find("#include \"") == 0)
		{
			// Include paths are relative to the base shader directory
			std::string include_path = line.substr(10);
			size_t      last_quote   = include_path.find("\"");
			if (!include_path.empty() && last_quote != std::string::npos)
			{
				include_path = include_path.substr(0, last_quote);
			}

			std::vector<uint8_t> raw_data;
			if (FileSystem::isExist(include_dir + include_path))
			{
				FileSystem::read(include_dir + include_path, raw_data);
			}
			std::string str;
			str.resize(raw_data.size());
			std::memcpy(str.data(), raw_data.data(), raw_data.size());

			auto include_file = precompile_shader(str, include_dir + FileSystem::getFileDirectory(include_path));
			for (auto &include_file_line : include_file)
			{
				include_file_line.erase(std::remove(include_file_line.begin(), include_file_line.end(), '\0'), include_file_line.end());
				final_file.push_back(include_file_line);
			}
		}
		else
		{
			final_file.push_back(line);
		}
	}

	return final_file;
}

ShaderCache::ShaderCache()
{
	ShaderCompiler::init();
}

ShaderCache::~ShaderCache()
{
	for (auto &shader_module : m_shader_modules)
	{
		if (shader_module != VK_NULL_HANDLE)
		{
			vkDestroyShaderModule(GraphicsContext::instance()->getLogicalDevice(), shader_module, nullptr);
		}
	}

	m_shader_modules.clear();

	ShaderCompiler::destroy();
}

VkShaderModule ShaderCache::load(const std::string &filename, VkShaderStageFlagBits stage, Shader::Type type, const std::string &entry_point, const std::vector<std::string> &macros)
{
	size_t define_hash = 0;
	for (auto &macro : macros)
	{
		hash_combine(define_hash, macro);
	}

	// Look for shader module
	if (m_lookup.find(filename + entry_point + std::to_string(define_hash)) != m_lookup.end())
	{
		return m_shader_modules.at(m_lookup[filename + entry_point + std::to_string(define_hash)]);
	}

	VK_INFO("Loading Shader {}", filename);

	std::vector<uint8_t> raw_data;
	FileSystem::read(filename, raw_data, type == Shader::Type::SPIRV);

	if (type == Shader::Type::GLSL)
	{
		// Convert to string
		std::string glsl_string;
		glsl_string.resize(raw_data.size());
		std::memcpy(glsl_string.data(), raw_data.data(), raw_data.size());
		auto glsl_strings = precompile_shader(glsl_string, FileSystem::getFileDirectory(filename));
		glsl_string.clear();
		for (auto &s : glsl_strings)
		{
			glsl_string += s + "\n";
		}
		raw_data.resize(glsl_string.size());
		std::memcpy(raw_data.data(), glsl_string.data(), glsl_string.size());
	}

	std::vector<uint32_t> spirv;

	if (type == Shader::Type::SPIRV)
	{
		spirv.resize(raw_data.size() / 4);
		std::memcpy(spirv.data(), raw_data.data(), raw_data.size());
	}
	else
	{
		spirv = ShaderCompiler::compile(filename, raw_data, stage, type, entry_point, macros);

		std::vector<uint8_t> write_data(spirv.size() * 4);
		std::memcpy(write_data.data(), spirv.data(), write_data.size());

		std::string spv_path = "Shader/" + FileSystem::getFileName(filename, false) + "_" + std::to_string(stage) + "_" + entry_point + "_" + std::to_string(define_hash) + ".spv ";
		
		FileSystem::save(spv_path, write_data, true);
	}

	m_reflection_data.emplace_back(std::move(ShaderReflection::reflect(spirv, stage)));

	VkShaderModuleCreateInfo shader_module_create_info = {};
	shader_module_create_info.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shader_module_create_info.codeSize                 = spirv.size() * sizeof(uint32_t);
	shader_module_create_info.pCode                    = spirv.data();

	VkShaderModule shader_module;
	if (!VK_CHECK(vkCreateShaderModule(GraphicsContext::instance()->getLogicalDevice(), &shader_module_create_info, nullptr, &shader_module)))
	{
		VK_ERROR("Failed to create shader module");
		return VK_NULL_HANDLE;
	}

	m_shader_modules.push_back(shader_module);
	m_lookup[filename + entry_point + std::to_string(define_hash)] = m_shader_modules.size() - 1;
	m_mapping[shader_module] = m_shader_modules.size() - 1;

	return shader_module;
}

VkShaderModule ShaderCache::getShader(const std::string &filename)
{
	if (m_lookup.find(filename) != m_lookup.end())
	{
		return m_shader_modules.at(m_lookup[filename]);
	}

	return VK_NULL_HANDLE;
}

const ReflectionData &ShaderCache::reflect(VkShaderModule shader)
{
	ASSERT(m_mapping.find(shader) != m_mapping.end());
	return m_reflection_data.at(m_mapping.at(shader));
}

const std::unordered_map<std::string, size_t> &ShaderCache::getShaders() const
{
	return m_lookup;
}
}        // namespace Ilum