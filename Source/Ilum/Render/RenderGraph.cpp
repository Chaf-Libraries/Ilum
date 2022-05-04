#include "RenderGraph.hpp"

#include "Pass/Triangle.hpp"

#include <imgui.h>
#include <imnodes.h>

#include <rttr/registration.h>

namespace Ilum
{
std::vector<std::string> RenderGraph::s_avaliable_passes = {
    "TrianglePass"};

RGResourceHandle::RGResourceHandle():
    m_index(CURRENT_ID++)
{
}

RGResourceHandle::operator uint32_t() const
{
	return m_index;
}

void RGResourceHandle::Invalidate()
{
	m_index = INVALID_ID;
}

bool RGResourceHandle::IsInvalid() const
{
	return m_index != INVALID_ID;
}

RGPassBuilder::RGPassBuilder(RGPass &pass, RenderGraph &graph) :
    m_pass(pass), m_graph(graph)
{
}

void RGPassBuilder::Bind(std::function<void(CommandBuffer &, const RGPassResources &)> &&callback)
{
}

RGResourceHandle RGPassBuilder::Write(RGResourceHandle &resource)
{
	return RGResourceHandle();
}

RGResourceHandle RGPassBuilder::CreateTexture(const std::string &name, const TextureDesc &desc)
{
	return m_graph.CreateTexture(name, desc);
}

RGResourceHandle RGPassBuilder::CreateBuffer(const std::string &name, const BufferDesc &desc)
{
	return RGResourceHandle();
}

RGPass::RGPass(RenderGraph &graph, const std::string &name) :
    m_graph(graph), m_name(name)
{
}

void RGPass::Execute(CommandBuffer &cmd_buffer, const RGPassResources &resource)
{
}

void RGPass::SetCallback(std::function<void(CommandBuffer &, const RGPassResources &)> &&callback)
{
}

bool RGPass::ReadFrom(RGResourceHandle handle) const
{
	return false;
}

bool RGPass::WriteTo(RGResourceHandle handle) const
{
	return false;
}

const std::string &RGPass::GetName() const
{
	return m_name;
}

RGPassResources::RGPassResources(RGPass &pass, RenderGraph &graph):
    m_pass(pass), m_graph(graph)
{
}

RGPassResources::~RGPassResources()
{
}

//Texture &RGPassResources::GetTexture(RGResourceHandle handle) const
//{
//	
//}
//
//Buffer &RGPassResources::GetBuffer(RGResourceHandle handle) const
//{
//	// TODO: 在此处插入 return 语句
//}

RenderGraph::RenderGraph(RHIDevice *device) :
    p_device(device)
{
}

RenderGraph::~RenderGraph()
{
}

RGPassBuilder RenderGraph::AddPass(const std::string &name)
{
	m_passes.emplace_back(std::make_unique<RGPass>(*this, name));
	return RGPassBuilder(*m_passes.back(), *this);
}

RGResourceHandle RenderGraph::CreateTexture(const std::string &name, const TextureDesc &desc)
{
	m_resources.emplace_back(std::make_unique<RGTexture>(name, desc));
	return CreateResourceNode(m_resources.back().get());
}

RGResourceHandle RenderGraph::CreateBuffer(const std::string &name, const BufferDesc &desc)
{
	return RGResourceHandle();
}

void RenderGraph::OnImGui()
{
	ImGui::Begin("Render Graph Editor");
	ImNodes::BeginNodeEditor();

	if (ImGui::BeginPopupContextWindow(0, 1, false))
	{
		ImGui::MenuItem("Add Pass", NULL, false, false);

		for (auto &avaliable_pass : s_avaliable_passes)
		{
			if (ImGui::MenuItem(avaliable_pass.c_str()))
			{
				rttr::variant pass_builder = rttr::type::get_by_name(avaliable_pass.c_str()).create();
				rttr::method  meth         = rttr::type::get_by_name(avaliable_pass.c_str()).get_method("BuildPass");
				meth.invoke(pass_builder, *this);
			}
		}

		ImGui::EndPopup();
	}

	// Draw Pass
	for (auto& pass : m_passes)
	{
		std::hash<void*> hasher;
		ImNodes::BeginNode(static_cast<int32_t>(hasher(pass.get())));

		ImGui::PushID(1);
		const int output_attr_id = 2;
		ImNodes::BeginOutputAttribute(output_attr_id);
		// in between Begin|EndAttribute calls, you can call ImGui
		// UI functions
		ImGui::Text("output pin");
		ImNodes::EndOutputAttribute();
		ImGui::PopID();
		ImNodes::EndNode();
	}

	ImNodes::MiniMap();
	ImNodes::EndNodeEditor();
	ImGui::End();
}

RGResourceHandle RenderGraph::CreateResourceNode(RGResource *resource)
{
	m_nodes.push_back(RGNode(resource));
	return RGResourceHandle();
}

}        // namespace Ilum