#include "RenderGraph.hpp"
#include "Renderer.hpp"

#include <RHI/Device.hpp>

#include <imgui.h>

namespace Ilum
{
RGPass::RGPass(RHIDevice *device, const std::string &name) :
    p_device(device), m_name(name)
{
}

RGPass::~RGPass()
{
}

void RGPass::Execute(CommandBuffer &cmd_buffer, const RGResources &resources, Renderer &renderer)
{
	if (!m_begin)
	{
		m_barrier_initialize(cmd_buffer);
		m_begin = true;
	}
	else
	{
		m_barrier_callback(cmd_buffer);
	}
	if (m_execute_callback)
	{
		m_execute_callback(cmd_buffer, m_pso, resources, renderer);
	}
}

void RGPass::OnImGui(ImGuiContext &context, const RGResources &resources)
{
	if (m_imgui_callback)
	{
		m_imgui_callback(context, resources);
	}
}

const std::string &RGPass::GetName() const
{
	return m_name;
}

RGNode::RGNode(RenderGraph &graph, RGPass &pass, RGResource *resource) :
    m_graph(graph), m_pass(pass), p_resource(resource)
{
}

RGResource *RGNode::GetResource()
{
	return p_resource;
}

const TextureState &RGNode::GetCurrentState() const
{
	return m_current_state;
}

const TextureState &RGNode::GetLastState() const
{
	return m_last_state;
}

RenderGraph::RenderGraph(RHIDevice *device, Renderer &renderer) :
    p_device(device), m_renderer(renderer)
{
}

RenderGraph ::~RenderGraph()
{
	m_passes.clear();
	m_nodes.clear();
	m_resources.clear();
}

void RenderGraph::Execute()
{
	for (auto &pass : m_passes)
	{
		auto &cmd_buffer = p_device->RequestCommandBuffer();
		cmd_buffer.Begin();
		auto resources = RGResources(*this, pass);
		pass.Execute(cmd_buffer, resources, m_renderer);
		cmd_buffer.End();
		p_device->Submit(cmd_buffer);
	}
}

Texture *RenderGraph::GetPresent() const
{
	return nullptr;
}

void RenderGraph::OnImGui(ImGuiContext &context)
{
	int32_t current_id = 0;
	for (auto &pass : m_passes)
	{
		ImGui::PushID(current_id++);
		if (ImGui::TreeNode(pass.GetName().c_str()))
		{
			auto resources = RGResources(*this, pass);
			pass.OnImGui(context, resources);
			ImGui::TreePop();
		}
		ImGui::PopID();
	}
}

RGResources::RGResources(RenderGraph &graph, RGPass &pass) :
    m_graph(graph), m_pass(pass)
{
}

Texture *RGResources::GetTexture(const RGHandle &handle) const
{
	auto &node = m_graph.m_nodes[handle];
	ASSERT(node->GetResource()->GetType() == ResourceType::Texture);
	return static_cast<RGTexture *>(node->GetResource())->GetHandle();
}

Buffer *RGResources::GetBuffer(const RGHandle &handle) const
{
	auto &node = m_graph.m_nodes[handle];
	ASSERT(node->GetResource()->GetType() == ResourceType::Buffer);
	return static_cast<RGBuffer *>(node->GetResource())->GetHandle();
}

}        // namespace Ilum