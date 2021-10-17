#include "Renderer.hpp"
#include "RenderGraph/RenderGraph.hpp"

#include "Renderer/RenderPass/ImGuiPass.hpp"

#include "Device/Window.hpp"
#include "Device/Swapchain.hpp"

#include "Graphics/GraphicsContext.hpp"

#include <imgui.h>

namespace Ilum
{
VkExtent2D Renderer::RenderTargetSize = {0, 0};

Renderer::Renderer(Context *context) :
    TSubsystem<Renderer>(context)
{
	GraphicsContext::instance()->Swapchain_Rebuild_Event += [this]() { rebuild(); };

	defaultBuilder = [this](RenderGraphBuilder &builder) {

	};

	buildRenderGraph = defaultBuilder;
}

Renderer::~Renderer()
{
}

bool Renderer::onInitialize()
{
	RenderTargetSize = GraphicsContext::instance()->getSwapchain().getExtent();

	defaultBuilder(m_rg_builder);

	m_render_graph = m_rg_builder.build();

	return true;
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

const RenderGraph &Renderer::getRenderGraph() const
{
	return *m_render_graph;
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
}
}        // namespace Ilum