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
	m_impl->data.Reset();

	for (auto &[node_handle, node] : m_impl->desc.GetNodes())
	{
		node.EmitHLSL(m_impl->desc, manager, &m_impl->context);
	}

	if (!m_impl->context.output.bsdf.empty())
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
			for (auto&bsdf: m_impl->context.bsdfs)
			{
				if (bsdf.name != m_impl->context.output.bsdf)
				{
					initializations << kainjow::mustache::data{"Initialization", fmt::format("{} {};", bsdf.type, bsdf.name)};
					initializations << kainjow::mustache::data{"Initialization", fmt::format("{}", bsdf.initialization)};
				}
				else
				{
					mustache_data.set("BxDFType", bsdf.type);
					mustache_data.set("BxDFName", bsdf.name);
				}
			}
			initializations << kainjow::mustache::data{"Initialization", m_impl->context.bsdfs.back().initialization};

			mustache_data.set("Initializations", initializations);
			mustache_data.set("Textures", textures);
			mustache_data.set("Samplers", samplers);
		}

		shader = mustache.render(mustache_data);
		shader = std::string(shader.c_str());

		m_impl->data.signature = fmt::format("Signature_{}", Hash(shader));

		shader_data.resize(shader.length());
		std::memcpy(shader_data.data(), shader.data(), shader_data.size());

		m_impl->data.shader = fmt::format("{}.material.hlsli", m_impl->desc.GetName());

		Path::GetInstance().Save(fmt::format("Asset/Material/{}", m_impl->data.shader), shader_data);

		// Update samplers
		for (auto &[sampler, desc] : m_impl->context.samplers)
		{
			m_impl->data.samplers.push_back(rhi_context->GetSamplerIndex(desc));
		}
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
			m_impl->data.textures.push_back(static_cast<uint32_t>(manager->Index<ResourceType::Texture2D>(texture_name)));
		}

		if (!m_impl->data.uniform_buffer || m_impl->data.uniform_buffer->GetDesc().size != (m_impl->data.textures.size() + m_impl->data.samplers.size()) * sizeof(uint32_t))
		{
			m_impl->data.uniform_buffer = rhi_context->CreateBuffer<uint32_t>(m_impl->data.textures.size() + m_impl->data.samplers.size(), RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);
		}

		std::vector<uint32_t> buffer_data = m_impl->data.textures;
		buffer_data.insert(buffer_data.end(), m_impl->data.samplers.begin(), m_impl->data.samplers.end());

		m_impl->data.uniform_buffer->CopyToDevice(buffer_data.data(), buffer_data.size() * sizeof(uint32_t));
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