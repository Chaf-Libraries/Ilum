#include "ShaderBuilder.hpp"
#include "ShaderCompiler.hpp"

#include <Core/Path.hpp>
#include <RHI/RHIDefinitions.hpp>

#include <filesystem>

namespace Ilum
{
struct ShaderBuilder::Impl
{
	std::unordered_map<size_t, std::unique_ptr<RHIShader>> shader_cache;
	std::unordered_map<RHIShader *, ShaderMeta> shader_meta_cache;
};

ShaderBuilder::ShaderBuilder(RHIContext *context) :
    p_rhi_context(context)
{
	m_impl = new Impl;
}

ShaderBuilder::~ShaderBuilder()
{
	p_rhi_context->WaitIdle();
	delete m_impl;
}

RHIShader *ShaderBuilder::RequireShader(const std::string &filename, const std::string &entry_point, RHIShaderStage stage, std::vector<std::string> &&macros, std::vector<std::string> &&includes, bool cuda, bool force_recompile)
{
	size_t hash = Hash(filename, entry_point, stage, macros, includes, cuda);

	if (!Path::GetInstance().IsExist("./bin/Shaders"))
	{
		Path::GetInstance().CreatePath("./bin/Shaders");
	}

	if (m_impl->shader_cache.find(hash) != m_impl->shader_cache.end() && !force_recompile)
	{
		return m_impl->shader_cache.at(hash).get();
	}

	std::string cache_path = "./bin/Shaders/" + std::to_string(hash) + ".shader";

	std::vector<uint8_t> shader_bin;
	ShaderMeta           meta;

	if (Path::GetInstance().IsExist(cache_path) && !force_recompile)
	{
		// Read from cache
		size_t last_write = 0;

		DESERIALIZE(
		    cache_path,
		    last_write,
		    shader_bin,
		    meta);

		if (last_write == std::filesystem::last_write_time(filename).time_since_epoch().count() && !shader_bin.empty())
		{
			LOG_INFO("Load shader {} with entry point \"{}\" from cache", filename, entry_point);
			std::unique_ptr<RHIShader> shader = p_rhi_context->CreateShader(entry_point, shader_bin, cuda);
			m_impl->shader_meta_cache.emplace(shader.get(), std::move(meta));
			m_impl->shader_cache.emplace(hash, std::move(shader));
			return m_impl->shader_cache.at(hash).get();
		}
		LOG_INFO("Cache of shader {} with entry point \"{}\" is out of date, recompile it", filename, entry_point);
	}

	{
		std::vector<uint8_t> shader_code;
		Path::GetInstance().Read(filename, shader_code);

		ShaderDesc desc = {};
		desc.code.resize(shader_code.size());
		std::memcpy(desc.code.data(), shader_code.data(), shader_code.size());
		for (auto &include : includes)
		{
			desc.code = fmt::format("#include \"{}\"\n", include) + desc.code;
		}

		desc.source      = Path::GetInstance().GetFileExtension(filename) == ".hlsl" ? ShaderSource::HLSL : ShaderSource::GLSL;
		desc.stage       = stage;
		desc.entry_point = entry_point;
		desc.macros      = macros;

		if (cuda)
		{
			desc.target = ShaderTarget::PTX;
		}
		else
		{
			if (p_rhi_context->GetBackend() == "Vulkan")
			{
				desc.target = ShaderTarget::SPIRV;
			}
			else if (p_rhi_context->GetBackend() == "DX12")
			{
				desc.target = ShaderTarget::DXIL;
			}
		}

		LOG_INFO("Compiling shader {} with entry point \"{}\"...", filename, entry_point);
		shader_bin = ShaderCompiler::GetInstance().Compile(desc, meta);

		if (shader_bin.empty())
		{
			LOG_ERROR("Shader {} with entry point \"{}\" compiled failed!", filename, entry_point);
			return nullptr;
		}

		LOG_ERROR("Shader {} with entry point \"{}\" compiled successfully, caching it...", filename, entry_point);

		SERIALIZE(
		    cache_path,
		    (size_t) std::filesystem::last_write_time(filename).time_since_epoch().count(),
		    shader_bin,
		    meta);

		std::unique_ptr<RHIShader> shader = p_rhi_context->CreateShader(entry_point, shader_bin, cuda);
		m_impl->shader_meta_cache.emplace(shader.get(), std::move(meta));
		m_impl->shader_cache.emplace(hash, std::move(shader));
		return m_impl->shader_cache.at(hash).get();
	}

	return nullptr;
}

ShaderMeta ShaderBuilder::RequireShaderMeta(RHIShader *shader) const
{
	return m_impl->shader_meta_cache.at(shader);
}
}        // namespace Ilum