#include "ShaderCache.hpp"
#include "GLSLCompiler.hpp"

#include <Core/FileSystem.hpp>
#include <Core/Hash.hpp>

namespace Ilum::Graphics
{
inline std::vector<std::string> PrecompileShader(const std::string &source, const std::string &include_dir)
{
	std::vector<std::string> final_file;

	auto lines = Core::FileSystem::Split(source, '\n');

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
			Core::FileSystem::Read(include_dir + include_path, raw_data);
			std::string str;
			str.resize(raw_data.size());
			std::memcpy(str.data(), raw_data.data(), raw_data.size());

			auto include_file = PrecompileShader(str, include_dir);
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

ShaderCache::ShaderCache(const Device &device) :
    m_device(device)
{
}

const Shader &ShaderCache::RequestShader(const std::string &path, VkShaderStageFlagBits stage)
{
	size_t hash = 0;
	Core::HashCombine(hash, path);
	Core::HashCombine(hash, static_cast<size_t>(stage));

	if (m_shaders.find(hash) != m_shaders.end())
	{
		return *m_shaders[hash];
	}

	std::string name = Core::FileSystem::GetFileName(path, false);

	// Check for spirv result
	std::string           spirv_path = "shaders/" + name + ".spv";
	std::vector<uint32_t> spirv;

	if (Core::FileSystem::IsExist(spirv_path))
	{
		VK_INFO("Loading spirv shader: {}", spirv_path);

		std::vector<uint8_t> raw_spirv;
		Core::FileSystem::Read(spirv_path, raw_spirv, true);
		spirv.resize(raw_spirv.size() / 4);
		std::memcpy(spirv.data(), raw_spirv.data(), raw_spirv.size());
	}
	else
	{
		VK_INFO("spirv shader {} is not found, compiling {}", spirv_path, path);

		std::vector<uint8_t> raw_glsl;
		Core::FileSystem::Read(path, raw_glsl);

		// Preprocess glsl header
		std::string glsl_string;
		glsl_string.resize(raw_glsl.size());
		std::memcpy(glsl_string.data(), raw_glsl.data(), raw_glsl.size());
		auto glsl_strings = PrecompileShader(glsl_string, Core::FileSystem::GetFileDirectory(path));
		glsl_string.clear();
		for (auto &s : glsl_strings)
		{
			glsl_string += s + "\n";
		}
		raw_glsl.resize(glsl_string.size());
		std::memcpy(raw_glsl.data(), glsl_string.data(), glsl_string.size());

		spirv = GLSLCompiler::Compile(raw_glsl, stage);

		std::vector<uint8_t> raw_spirv;
		raw_spirv.resize(spirv.size() * 4);
		std::memcpy(raw_spirv.data(), spirv.data(), raw_spirv.size());

		if (!Core::FileSystem::IsExist("shaders/"))
		{
			Core::FileSystem::CreatePath("shaders/");
		}
		Core::FileSystem::Save(spirv_path, raw_spirv, true);
	}

	m_shaders.emplace(hash, std::make_unique<Shader>(m_device, spirv, stage));
	return *m_shaders[hash];
}
}        // namespace Ilum::Graphics