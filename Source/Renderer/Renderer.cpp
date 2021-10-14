#include "Renderer.hpp"
#include "RenderGraph/RenderGraph.hpp"

#include "Renderer/RenderPass/ImGuiPass.hpp"

#include "Device/Window.hpp"
#include "Device/Swapchain.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Graphics/ImGui/imgui.h"

namespace Ilum
{
Renderer::Renderer(Context *context) :
    TSubsystem<Renderer>(context)
{
	GraphicsContext::instance()->Swapchain_Rebuild_Event += [this]() { rebuild(); };

	defaultBuilder = [this](RenderGraphBuilder &builder) {
		builder.addRenderPass("ImGuiPass", std::make_unique<ImGuiPass>("output")).setOutput("output");
	};
}

Renderer::~Renderer()
{
}

bool Renderer::onInitialize()
{
	defaultBuilder(m_rg_builder);

	m_render_graph = m_rg_builder.build();

	return true;
}

void Renderer::onPostTick()
{
	m_render_graph->execute(GraphicsContext::instance()->getCurrentCommandBuffer());
	m_render_graph->present(GraphicsContext::instance()->getCurrentCommandBuffer(), GraphicsContext::instance()->getSwapchain().getImages()[GraphicsContext::instance()->getFrameIndex()]);
}

RenderGraphBuilder &Renderer::getRenderGraphBuilder()
{
	return m_rg_builder;
}

const RenderGraph &Renderer::getRenderGraph() const
{
	return *m_render_graph;
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