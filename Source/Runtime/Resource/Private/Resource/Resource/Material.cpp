#include "Resource/Material.hpp"

#include <Material/MaterialCompiler.hpp>

#include <mustache.hpp>

namespace Ilum
{
struct Resource<ResourceType::Material>::Impl
{
	MaterialGraphDesc desc;
	std::string       layout;

	bool valid = false;
};

Resource<ResourceType::Material>::Resource(RHIContext *rhi_context, const std::string &name) :
    IResource(rhi_context, name, ResourceType::Material)
{
}

Resource<ResourceType::Material>::Resource(RHIContext *rhi_context, const std::string &name, MaterialGraphDesc &&desc) :
    IResource(name)
{
	m_impl = new Impl;

	m_impl->desc = std::move(desc);

	std::vector<uint8_t> thumbnail_data;

	SERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Material), thumbnail_data, m_impl->desc, m_impl->layout);
}

Resource<ResourceType::Material>::~Resource()
{
	delete m_impl;
}

bool Resource<ResourceType::Material>::Validate() const
{
	return m_impl != nullptr;
}

void Resource<ResourceType::Material>::Load(RHIContext *rhi_context)
{
	m_impl = new Impl;

	std::vector<uint8_t> thumbnail_data;
	DESERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Material), thumbnail_data, m_impl->desc, m_impl->layout);
}

void Resource<ResourceType::Material>::Compile(Renderer *renderer, const std::string &layout)
{
	m_impl->layout = layout;
	m_impl->valid  = false;

	MaterialCompilationContext context;

	for (auto &[node_handle, node] : m_impl->desc.GetNodes())
	{
		node.EmitHLSL(m_impl->desc, renderer, &context);
	}

	if (!context.bsdfs.empty())
	{
		std::vector<uint8_t> shader_data;
		Path::GetInstance().Read("Source/Shaders/Material.hlsli", shader_data);
		std::string shader(shader_data.begin(), shader_data.end());

		kainjow::mustache::mustache mustache = {shader};
		kainjow::mustache::data     mustache_data{kainjow::mustache::data::type::object};

		{
			kainjow::mustache::data initializations{kainjow::mustache::data::type::list};
			kainjow::mustache::data textures{kainjow::mustache::data::type::list};
			kainjow::mustache::data samplers{kainjow::mustache::data::type::list};
			for (auto &variable : context.variables)
			{
				initializations << kainjow::mustache::data{"Initialization", variable};
			}
			for (auto &[texture, texture_name] : context.textures)
			{
				textures << kainjow::mustache::data{"Texture", texture};
			}
			for (auto &[sampler, desc] : context.samplers)
			{
				samplers << kainjow::mustache::data{"Sampler", sampler};
			}
			for (uint32_t i = 0; i < context.bsdfs.size() - 1; i++)
			{
				initializations << kainjow::mustache::data{"Initialization", fmt::format("{} {}", context.bsdfs[i].type, context.bsdfs[i].initialization)};
			}
			initializations << kainjow::mustache::data{"Initialization", context.bsdfs.back().initialization};

			mustache_data.set("BxDFType", context.bsdfs.back().type);
			mustache_data.set("BxDFName", context.bsdfs.back().name);
			mustache_data.set("Initializations", initializations);
			mustache_data.set("Textures", textures);
			mustache_data.set("Samplers", samplers);
		}

		shader = mustache.render(mustache_data);

		shader_data.resize(shader.length());
		std::memcpy(shader_data.data(), shader.data(), shader_data.size());

		Path::GetInstance().Save(fmt::format("Asset/Material/{}.material.hlsli", m_impl->desc.GetName()), shader_data);
	}

	std::vector<uint8_t> thumbnail_data;

	SERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Material), thumbnail_data, m_impl->desc, m_impl->layout);

	m_impl->valid = true;
}

const std::string &Resource<ResourceType::Material>::GetLayout() const
{
	return m_impl->layout;
}

MaterialGraphDesc &Resource<ResourceType::Material>::GetDesc()
{
	return m_impl->desc;
}

bool Resource<ResourceType::Material>::IsValid() const
{
	return m_impl->valid;
}
}        // namespace Ilum