#include "Renderer.hpp"
#include "RenderGraph/RenderGraph.hpp"

#include "Renderer/RenderPass/ImGuiPass.hpp"

#include "Device/Swapchain.hpp"
#include "Device/Window.hpp"

#include "Graphics/GraphicsContext.hpp"

#include <imgui.h>

namespace Ilum
{
VkExtent2D Renderer::RenderTargetSize = {0, 0};

Renderer::Renderer(Context *context) :
    TSubsystem<Renderer>(context)
{
	GraphicsContext::instance()->Swapchain_Rebuild_Event += [this]() { m_resize = true; };

	defaultBuilder = [this](RenderGraphBuilder &builder) {

	};

	buildRenderGraph = defaultBuilder;
	m_resource_cache = createScope<ResourceCache>();
	createSamplers();
}

Renderer::~Renderer()
{
}

bool Renderer::onInitialize()
{
	RenderTargetSize = GraphicsContext::instance()->getSwapchain().getExtent();

	defaultBuilder(m_rg_builder);

	rebuild();

	return true;
}

void Renderer::onPreTick()
{
	if (m_resize)
	{
		rebuild();
		m_resize = false;
	}
}

void Renderer::onPostTick()
{
	if (!m_render_graph)
	{
		return;
	}

	m_render_graph->execute(GraphicsContext::instance()->getCurrentCommandBuffer());
	m_render_graph->present(GraphicsContext::instance()->getCurrentCommandBuffer(), GraphicsContext::instance()->getSwapchain().getImages()[GraphicsContext::instance()->getFrameIndex()]);
}

void Renderer::onShutdown()
{
	m_samplers.clear();
}

const RenderGraph &Renderer::getRenderGraph() const
{
	return *m_render_graph;
}

ResourceCache &Renderer::getResourceCache()
{
	return *m_resource_cache;
}

void Renderer::resetBuilder()
{
	buildRenderGraph = defaultBuilder;
}

void Renderer::rebuild()
{
	m_rg_builder.reset();

	if (buildRenderGraph)
	{
		buildRenderGraph(m_rg_builder);
	}
	else
	{
		defaultBuilder(m_rg_builder);
	}

	m_render_graph = m_rg_builder.build();
	Event_RenderGraph_Rebuild.invoke();
}

const Sampler &Renderer::getSampler(SamplerType type) const
{
	return m_samplers.at(type);
}

void Renderer::createSamplers()
{
	m_samplers[SamplerType::Compare_Depth]     = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_NEAREST);
	m_samplers[SamplerType::Point_Clamp]       = Sampler(VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_NEAREST);
	m_samplers[SamplerType::Point_Wrap]        = Sampler(VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_FILTER_NEAREST);
	m_samplers[SamplerType::Bilinear_Clamp]    = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_NEAREST);
	m_samplers[SamplerType::Bilinear_Wrap]     = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_FILTER_NEAREST);
	m_samplers[SamplerType::Trilinear_Clamp]   = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR);
	m_samplers[SamplerType::Trilinear_Wrap]    = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_FILTER_LINEAR);
	m_samplers[SamplerType::Anisptropic_Clamp] = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR);
	m_samplers[SamplerType::Anisptropic_Wrap]  = Sampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_FILTER_LINEAR);
}
}        // namespace Ilum