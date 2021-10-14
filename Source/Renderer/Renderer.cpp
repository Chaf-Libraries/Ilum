#include "Renderer.hpp"
#include "RenderGraph/RenderGraph.hpp"

#include "Renderer/RenderPass/ImGuiPass.hpp"

#include "Device/Window.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/RenderPass/Swapchain.hpp"

#include "Graphics/ImGui/imgui.h"

namespace Ilum
{
Renderer::Renderer(Context *context) :
    TSubsystem<Renderer>(context)
{
}

Renderer::~Renderer()
{
}

bool Renderer::onInitialize()
{
	GraphicsContext::instance()->Swapchain_Rebuild_Event += [this]() { m_render_graph = m_rg_builder.build(); };

	m_rg_builder.addRenderPass("ImGuiPass", std::make_unique<ImGuiPass>("output")).setOutput("output");
	m_render_graph = m_rg_builder.build();

	return true;
}

void Renderer::onTick(float)
{
	if (!m_render_graph)
	{
		m_render_graph = m_rg_builder.build();
	}

	// Begin docking space
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
	window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
	window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
	ImGuiViewport *viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(viewport->WorkPos);
	ImGui::SetNextWindowSize(viewport->WorkSize);
	ImGui::SetNextWindowViewport(viewport->ID);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::Begin("DockSpace", (bool *) 1, window_flags);
	ImGui::PopStyleVar();
	ImGui::PopStyleVar(2);

	ImGuiIO &io = ImGui::GetIO();
	if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
	{
		ImGuiID dockspace_id = ImGui::GetID("DockSpace");
		ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
	}
	ImGui::ShowDemoWindow();
	// End docking space
	ImGui::End();

	io.DisplaySize = ImVec2(static_cast<float>(Window::instance()->getWidth()), static_cast<float>(Window::instance()->getHeight()));

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
	m_render_graph = m_rg_builder.build();
}
}        // namespace Ilum