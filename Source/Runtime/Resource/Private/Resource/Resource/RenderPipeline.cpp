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
	m_impl = new Impl;
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

std::unique_ptr<RenderGraph> Resource<ResourceType::RenderPipeline>::Compile(RHIContext *rhi_context, Renderer *renderer, const std::string &layout)
{
	m_impl->layout = layout;
	
	RenderGraphBuilder builder(rhi_context);
	auto render_graph = builder.Compile(m_impl->desc, renderer);

	std::vector<uint32_t> thumbnail_data;
	SERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::RenderPipeline), thumbnail_data, m_impl->desc, m_impl->layout);

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
