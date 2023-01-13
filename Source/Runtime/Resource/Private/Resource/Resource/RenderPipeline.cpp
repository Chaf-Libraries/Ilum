#include "Resource/RenderPipeline.hpp"

#include <RenderGraph/RenderGraph.hpp>
#include <RenderGraph/RenderGraphBuilder.hpp>

namespace Ilum
{
struct Resource<ResourceType::RenderPipeline>::Impl
{
	RenderGraphDesc desc;
	std::string     layout;
};

Resource<ResourceType::RenderPipeline>::Resource(RHIContext *rhi_context, const std::string &name) :
    IResource(rhi_context, name, ResourceType::RenderPipeline)
{
}

Resource<ResourceType::RenderPipeline>::Resource(RHIContext *rhi_context, const std::string &name, RenderGraphDesc &&desc) :
    IResource(name)
{
	m_impl       = new Impl;
	m_impl->desc = std::move(desc);

	std::vector<uint32_t> thumbnail_data;
	SERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", name, (uint32_t) ResourceType::RenderPipeline), thumbnail_data, m_impl->desc, m_impl->layout);
}

Resource<ResourceType::RenderPipeline>::~Resource()
{
	delete m_impl;
}

bool Resource<ResourceType::RenderPipeline>::Validate() const
{
	return m_impl != nullptr;
}

void Resource<ResourceType::RenderPipeline>::Load(RHIContext *rhi_context)
{
	m_impl = new Impl;

	std::vector<uint32_t> thumbnail_data;
	DESERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::RenderPipeline), thumbnail_data, m_impl->desc, m_impl->layout);
}

std::unique_ptr<RenderGraph> Resource<ResourceType::RenderPipeline>::Compile(RHIContext *rhi_context, Renderer *renderer, glm::vec2 viewport, const std::string &layout)
{
	m_impl->layout = layout;

	std::vector<uint32_t> thumbnail_data;
	SERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::RenderPipeline), thumbnail_data, m_impl->desc, m_impl->layout);

	RenderGraphDesc desc = m_impl->desc;

	for (auto &[pass_handle, pass] : desc.GetPasses())
	{
		for (auto &[pin_handle, pin] : pass.GetPins())
		{
			if (pin.attribute == RenderPassPin::Attribute::Output &&
			    pin.type == RenderPassPin::Type::Texture &&
			    (pin.texture.width == 0 || pin.texture.height == 0))
			{
				if (viewport.x <= 0 || viewport.y <= 0)
				{
					return nullptr;
				}
				pin.texture.width = static_cast<uint32_t>(viewport.x);
				pin.texture.height = static_cast<uint32_t>(viewport.y);
			}
		}
	}

	RenderGraphBuilder builder(rhi_context);
	auto               render_graph = builder.Compile(desc, renderer);

	
	return render_graph;
}

const std::string &Resource<ResourceType::RenderPipeline>::GetLayout() const
{
	return m_impl->layout;
}

RenderGraphDesc &Resource<ResourceType::RenderPipeline>::GetDesc()
{
	return m_impl->desc;
}
}        // namespace Ilum
