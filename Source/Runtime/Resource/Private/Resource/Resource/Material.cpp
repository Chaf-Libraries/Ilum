#include "Resource/Material.hpp"
#include "Resource/Texture2D.hpp"
#include "ResourceManager.hpp"

#include <Material/MaterialCompiler.hpp>
#include <Material/MaterialData.hpp>

#include <mustache.hpp>

namespace Ilum
{
struct Resource<ResourceType::Material>::Impl
{
	MaterialGraphDesc desc;

	MaterialCompilationContext context;

	std::string layout;

	MaterialData data;

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

void Resource<ResourceType::Material>::Compile(RHIContext *rhi_context, ResourceManager *manager, RHITexture *dummy_texture, const std::string &layout)
{
	if (!layout.empty())
	{
		m_impl->layout = layout;
	}

	m_impl->valid = false;

	m_impl->context.Reset();

	for (auto &[node_handle, node] : m_impl->desc.GetNodes())
	{
		node.EmitHLSL(m_impl->desc, manager, &m_impl->context);
	}

	if (!m_impl->context.bsdfs.empty())
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
			for (auto &variable : m_impl->context.variables)
			{
				initializations << kainjow::mustache::data{"Initialization", variable};
			}
			for (auto &[texture, texture_name] : m_impl->context.textures)
			{
				textures << kainjow::mustache::data{"Texture", texture};
			}
			for (auto &[sampler, desc] : m_impl->context.samplers)
			{
				samplers << kainjow::mustache::data{"Sampler", sampler};
			}
			for (uint32_t i = 0; i < m_impl->context.bsdfs.size() - 1; i++)
			{
				initializations << kainjow::mustache::data{"Initialization", fmt::format("{} {}", m_impl->context.bsdfs[i].type, m_impl->context.bsdfs[i].initialization)};
			}
			initializations << kainjow::mustache::data{"Initialization", m_impl->context.bsdfs.back().initialization};

			mustache_data.set("BxDFType", m_impl->context.bsdfs.back().type);
			mustache_data.set("BxDFName", m_impl->context.bsdfs.back().name);
			mustache_data.set("Initializations", initializations);
			mustache_data.set("Textures", textures);
			mustache_data.set("Samplers", samplers);
		}

		shader = mustache.render(mustache_data);

		shader_data.resize(shader.length());
		std::memcpy(shader_data.data(), shader.data(), shader_data.size());

		m_impl->data.shader = fmt::format("Asset/Material/{}.material.hlsli", m_impl->desc.GetName());

		Path::GetInstance().Save(m_impl->data.shader, shader_data);

		// Update samplers
		for (auto &[sampler, desc] : m_impl->context.samplers)
		{
			m_impl->data.samplers[sampler] = rhi_context->CreateSampler(desc);
		}
	}
	else
	{
		m_impl->data.shader = "Asset/Material/default.material.hlsli";
	}

	std::vector<uint8_t> thumbnail_data;

	SERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Material), thumbnail_data, m_impl->desc, m_impl->layout);

	m_impl->valid = true;

	Update(rhi_context, manager, dummy_texture);
}

void Resource<ResourceType::Material>::Update(RHIContext *rhi_context, ResourceManager *manager, RHITexture *dummy_texture)
{
	if (!m_impl->valid)
	{
		Compile(rhi_context, manager, dummy_texture);
	}
	else
	{
		m_impl->data.textures.clear();
		for (auto &[texture, texture_name] : m_impl->context.textures)
		{
			m_impl->data.textures[texture] = manager->Has<ResourceType::Texture2D>(texture_name) ? manager->Get<ResourceType::Texture2D>(texture_name)->GetTexture() : dummy_texture;
		}
	}
}

const MaterialData &Resource<ResourceType::Material>::GetMaterialData() const
{
	return m_impl->data;
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