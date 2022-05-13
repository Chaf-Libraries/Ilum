#include "RGBuilder.hpp"
#include "RenderGraph.hpp"
#include "RenderPass.hpp"

#include <RHI/Device.hpp>
#include <RHI/ImGuiContext.hpp>

#include <Core/Path.hpp>

#include <imgui.h>
#include <imnodes.h>

#include <rttr/registration.h>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include <fstream>

namespace Ilum
{
std::vector<std::string> RGBuilder::s_avaliable_passes = {
    "VBuffer",
    "VisualizeVBuffer",
    "Triangle",
    "SkyboxPass",
    "Present"};

inline bool IsRead(VkAccessFlags access)
{
	if ((access & VK_ACCESS_SHADER_WRITE_BIT) ||
	    (access & VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT) ||
	    (access & VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) ||
	    (access & VK_ACCESS_TRANSFER_WRITE_BIT) ||
	    (access & VK_ACCESS_HOST_WRITE_BIT) ||
	    (access & VK_ACCESS_MEMORY_WRITE_BIT))
	{
		return false;
	}
	return true;
}

struct RGSerializeData
{
	std::vector<std::string>           passes;
	std::vector<uint32_t>              pass_handles;
	std::vector<std::vector<uint32_t>> nodes;
	std::map<uint32_t, uint32_t>       edges;

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(passes, pass_handles, nodes, edges);
	}
};

RGBuilder::RGBuilder(RHIDevice *device, RenderGraph &graph, Renderer &renderer) :
    p_device(device), m_graph(graph), m_renderer(renderer)
{
}

RGBuilder::~RGBuilder()
{
	m_render_passes.clear();
	m_resources.clear();
	m_edges.clear();
}

RGHandle RGBuilder::CreateTexture(const std::string &name, const TextureDesc &desc, const TextureState &state)
{
	auto handle = RGHandle();

	m_resources[handle] = std::make_unique<TextureDeclaration>(name, desc, state);
	return handle;
}

RGHandle RGBuilder::CreateBuffer(const std::string &name, const BufferDesc &desc, const BufferState &state)
{
	auto handle = RGHandle();

	m_resources[handle] = std::make_unique<BufferDeclaration>(name, desc, state);
	return handle;
}

RGBuilder &RGBuilder::AddPass(std::unique_ptr<RenderPass> &&pass)
{
	m_render_passes.emplace_back(std::move(pass));
	return *this;
}

RGBuilder &RGBuilder::Link(const RGHandle &from, const RGHandle &to)
{
	size_t hash = 0;
	HashCombine(hash, (uint32_t) from);
	HashCombine(hash, (uint32_t) to);

	if (m_edges.find(static_cast<uint32_t>(hash)) == m_edges.end())
	{
		if (m_resources[from]->GetType() == m_resources[to]->GetType())
		{
			// Link buffer
			if (m_resources[from]->GetType() == ResourceType::Buffer)
			{
				auto *src = static_cast<BufferDeclaration *>(m_resources[from].get());
				auto *dst = static_cast<BufferDeclaration *>(m_resources[to].get());
				if (!(IsRead(src->GetState().access_mask) && IsRead(dst->GetState().access_mask)))
				{
					if (src->GetDesc().size == dst->GetDesc().size &&
					    src->GetDesc().memory_usage == dst->GetDesc().memory_usage)
					{
						m_edges[static_cast<uint32_t>(hash)] = std::make_pair(from, to);
					}
				}
			}
			// Link Texture
			if (m_resources[from]->GetType() == ResourceType::Texture)
			{
				auto *src = static_cast<TextureDeclaration *>(m_resources[from].get());
				auto *dst = static_cast<TextureDeclaration *>(m_resources[to].get());
				if (!(IsRead(src->GetState().access_mask) && IsRead(dst->GetState().access_mask)))
				{
					m_edges[static_cast<uint32_t>(hash)] = std::make_pair(from, to);
				}
			}
		}
	}

	return *this;
}

void RGBuilder::Compile()
{
	p_device->Reset();

	m_graph.m_nodes.clear();
	m_graph.m_passes.clear();
	m_graph.m_resources.clear();

	std::vector<RenderPass *> passes;

	for (auto &pass : m_render_passes)
	{
		passes.push_back(pass.get());
	}

	std::map<uint32_t, std::pair<RGHandle, RGHandle>> edges = m_edges;
	std::vector<std::vector<TextureTransition>>       texture_transitions;
	std::vector<std::vector<BufferTransition>>        buffer_transitions;
	std::vector<std::vector<RGNode *>>                nodes;

	// Topology sort
	while (!passes.empty())
	{
		for (auto iter = passes.begin(); iter != passes.end();)
		{
			auto &resources = (*iter)->GetResources();
			bool  top       = true;
			for (auto &resource : resources)
			{
				for (auto &[hash, edge] : edges)
				{
					if (resource == edge.second)
					{
						top = false;
						break;
					}
				}
				if (!top)
				{
					break;
				}
			}
			if (top)
			{
				m_graph.m_passes.push_back(RGPass(p_device, (*iter)->GetName()));
				(*iter)->Build(m_graph.m_passes.back());
				nodes.push_back({});
				for (auto &resource : resources)
				{
					m_graph.m_nodes[resource] = std::make_unique<RGNode>(m_graph, m_graph.m_passes.back(), nullptr);
					nodes.back().push_back(m_graph.m_nodes[resource].get());
					// Set resource state
					if (m_resources[resource]->GetType() == ResourceType::Texture)
					{
						m_graph.m_nodes[resource]->m_current_state = static_cast<TextureDeclaration *>(m_resources[resource].get())->GetState();
					}
					else if (m_resources[resource]->GetType() == ResourceType::Buffer)
					{
						BufferState state                                      = static_cast<BufferDeclaration *>(m_resources[resource].get())->GetState();
						m_graph.m_nodes[resource]->m_current_state.access_mask = state.access_mask;
						m_graph.m_nodes[resource]->m_current_state.stage       = state.stage;
					}

					for (auto iter = edges.begin(); iter != edges.end();)
					{
						if (resource == iter->second.first)
						{
							iter = edges.erase(iter);
						}
						else
						{
							iter++;
						}
					}
				}

				iter = passes.erase(iter);
				break;
			}
			else
			{
				iter++;
			}
		}
	}

	// Resolve Resource
	edges = m_edges;
	for (auto &[handle, node] : m_graph.m_nodes)
	{
		if (node->p_resource)
		{
			continue;
		}
		// All nodes that shared one resource
		std::set<RGHandle> shares;
		 shares.insert(handle);
		 for (auto &[hash, edge] : edges)
		{
			 if (shares.find(edge.first) != shares.end() || 
				 shares.find(edge.second) != shares.end())
			{
				shares.insert(edge.first);
				shares.insert(edge.second);
			}
		 }
		//  Allocate new resource
		if (m_resources[handle]->GetType() == ResourceType::Texture)
		{
			TextureDesc desc = {};
			for (auto &share : shares)
			{
				auto *resource = static_cast<TextureDeclaration *>(m_resources[share].get());

				desc.width        = std::max(desc.width, resource->GetDesc().width);
				desc.height       = std::max(desc.height, resource->GetDesc().height);
				desc.depth        = std::max(desc.depth, resource->GetDesc().depth);
				desc.mips         = std::max(desc.mips, resource->GetDesc().mips);
				desc.layers       = std::max(desc.layers, resource->GetDesc().layers);
				desc.sample_count = std::max(desc.sample_count, resource->GetDesc().sample_count);
				desc.format       = std::max(desc.format, resource->GetDesc().format);
				desc.usage |= resource->GetDesc().usage;
			}
			m_graph.m_resources.emplace_back(std::make_unique<RGTexture>(p_device, desc));
			static_cast<RGTexture *>(m_graph.m_resources.back().get())->GetHandle()->SetName(m_resources.begin()->second->GetName());
		}
		else if (m_resources[handle]->GetType() == ResourceType::Buffer)
		{
			BufferDesc desc = {};
			for (auto &share : shares)
			{
				auto *resource = static_cast<BufferDeclaration *>(m_resources[share].get());

				desc.buffer_usage |= resource->GetDesc().buffer_usage;
				desc.memory_usage = std::max(desc.memory_usage, resource->GetDesc().memory_usage);
				desc.size         = std::max(desc.size, resource->GetDesc().size);
			}
			m_graph.m_resources.emplace_back(std::make_unique<RGBuffer>(p_device, desc));
			static_cast<RGBuffer *>(m_graph.m_resources.back().get())->GetHandle()->SetName(m_resources.begin()->second->GetName());
		}
		// Assign for all nodes
		for (auto &share : shares)
		{
			m_graph.m_nodes[share]->p_resource = m_graph.m_resources.back().get();
		}
	}

	// Bake resource transition
	std::unordered_map<RGNode *, TextureState>     resource_initial_state;
	std::unordered_map<RGResource *, TextureState> resource_state;
	for (size_t i = 0; i < m_graph.m_passes.size(); i++)
	{
		for (auto &node : nodes[i])
		{
			if (resource_state.find(node->GetResource()) != resource_state.end())
			{
				node->m_last_state                  = resource_state[node->GetResource()];
				resource_state[node->GetResource()] = node->m_current_state;
			}
			else
			{
				resource_state[node->GetResource()] = node->m_current_state;
				resource_initial_state[node]        = node->m_current_state;
			}
		}
	}

	// Reset initial state's last state
	for (auto &[node, state] : resource_initial_state)
	{
		if (resource_state.find(node->GetResource()) != resource_state.end())
		{
			node->m_last_state = resource_state[node->GetResource()];
		}
	}

	// Set Pipeline Barrier initiialize
	for (size_t i = 0; i < m_graph.m_passes.size(); i++)
	{
		auto &pass                = m_graph.m_passes[i];
		pass.m_barrier_initialize = [=](CommandBuffer &cmd_buffer) {
			std::vector<BufferTransition>  buffer_transitions;
			std::vector<TextureTransition> texture_transitions;
			for (auto &node : nodes[i])
			{
				if (resource_initial_state.find(node) != resource_initial_state.end())
				{
					if (node->GetResource()->GetType() == ResourceType::Buffer)
					{
						BufferState src = {};
						BufferState dst = {};

						dst.access_mask = resource_initial_state.at(node).access_mask;
						dst.stage       = resource_initial_state.at(node).stage;
						buffer_transitions.push_back(BufferTransition{static_cast<RGBuffer *>(node->GetResource())->GetHandle(), src, dst});
					}
					if (node->GetResource()->GetType() == ResourceType::Texture)
					{
						TextureState src = {};
						TextureState dst = {};

						dst.access_mask            = resource_initial_state.at(node).access_mask;
						dst.stage                  = resource_initial_state.at(node).stage;
						dst.layout                 = resource_initial_state.at(node).layout;
						Texture           *texture = static_cast<RGTexture *>(node->GetResource())->GetHandle();
						VkImageAspectFlags aspect  = texture->IsDepth() ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
						aspect |= texture->IsStencil() ? VK_IMAGE_ASPECT_STENCIL_BIT : 0;
						texture_transitions.push_back(TextureTransition{
						    texture,
						    src, dst,
						    VkImageSubresourceRange{aspect, 0, texture->GetMipLevels(), 0, texture->GetLayerCount()}});
					}
				}
				else
				{
					if (node->GetResource()->GetType() == ResourceType::Buffer)
					{
						BufferState src = {};
						BufferState dst = {};
						src.access_mask = node->GetLastState().access_mask;
						src.stage       = node->GetLastState().stage;
						dst.access_mask = node->GetCurrentState().access_mask;
						dst.stage       = node->GetCurrentState().stage;
						buffer_transitions.push_back(BufferTransition{static_cast<RGBuffer *>(node->GetResource())->GetHandle(), src, dst});
					}
					if (node->GetResource()->GetType() == ResourceType::Texture)
					{
						TextureState src           = {};
						TextureState dst           = {};
						src.access_mask            = node->GetLastState().access_mask;
						src.stage                  = node->GetLastState().stage;
						src.layout                 = node->GetLastState().layout;
						dst.access_mask            = node->GetCurrentState().access_mask;
						dst.stage                  = node->GetCurrentState().stage;
						dst.layout                 = node->GetCurrentState().layout;
						Texture           *texture = static_cast<RGTexture *>(node->GetResource())->GetHandle();
						VkImageAspectFlags aspect  = texture->IsDepth() ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
						aspect |= texture->IsStencil() ? VK_IMAGE_ASPECT_STENCIL_BIT : 0;
						texture_transitions.push_back(TextureTransition{
						    texture,
						    src, dst,
						    VkImageSubresourceRange{aspect, 0, texture->GetMipLevels(), 0, texture->GetLayerCount()}});
					}
				}
			}
			if (!buffer_transitions.empty() || !texture_transitions.empty())
			{
				cmd_buffer.Transition(buffer_transitions, texture_transitions);
			}
		};
	}

	// Insert pipeline barrier
	for (size_t i = 0; i < m_graph.m_passes.size(); i++)
	{
		auto &pass              = m_graph.m_passes[i];
		pass.m_barrier_callback = [=](CommandBuffer &cmd_buffer) {
			std::vector<BufferTransition>  buffer_transitions;
			std::vector<TextureTransition> texture_transitions;
			for (auto &node : nodes[i])
			{
				if (node->GetResource()->GetType() == ResourceType::Buffer)
				{
					if (node->GetLastState().access_mask != node->GetCurrentState().access_mask ||
					    node->GetLastState().stage != node->GetCurrentState().stage)
					{
						BufferState src = {};
						BufferState dst = {};
						src.access_mask = node->GetLastState().access_mask;
						src.stage       = node->GetLastState().stage;
						dst.access_mask = node->GetCurrentState().access_mask;
						dst.stage       = node->GetCurrentState().stage;
						buffer_transitions.push_back(BufferTransition{static_cast<RGBuffer *>(node->GetResource())->GetHandle(), src, dst});
					}
				}
				if (node->GetResource()->GetType() == ResourceType::Texture)
				{
					if (node->GetLastState().layout != node->GetCurrentState().layout ||
					    node->GetLastState().access_mask != node->GetCurrentState().access_mask ||
					    node->GetLastState().stage != node->GetCurrentState().stage)
					{
						TextureState src           = {};
						TextureState dst           = {};
						src.access_mask            = node->GetLastState().access_mask;
						src.stage                  = node->GetLastState().stage;
						src.layout                 = node->GetLastState().layout;
						dst.access_mask            = node->GetCurrentState().access_mask;
						dst.stage                  = node->GetCurrentState().stage;
						dst.layout                 = node->GetCurrentState().layout;
						Texture           *texture = static_cast<RGTexture *>(node->GetResource())->GetHandle();
						VkImageAspectFlags aspect  = texture->IsDepth() ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
						aspect |= texture->IsStencil() ? VK_IMAGE_ASPECT_STENCIL_BIT : 0;
						texture_transitions.push_back(TextureTransition{
						    texture,
						    src, dst,
						    VkImageSubresourceRange{aspect, 0, texture->GetMipLevels(), 0, texture->GetLayerCount()}});
					}
				}
			}
			if (!buffer_transitions.empty() || !texture_transitions.empty())
			{
				cmd_buffer.Transition(buffer_transitions, texture_transitions);
			}
		};
	}
}

bool RGBuilder::OnImGui(ImGuiContext &context)
{
	bool recompile = false;

	ImGui::Begin("Render Graph Editor");

	std::vector<int32_t> selected_links;
	std::vector<int32_t> selected_nodes;

	if (ImNodes::NumSelectedLinks() > 0)
	{
		selected_links.resize(ImNodes::NumSelectedLinks());
		ImNodes::GetSelectedLinks(selected_links.data());
	}

	if (ImNodes::NumSelectedNodes() > 0)
	{
		selected_nodes.resize(ImNodes::NumSelectedNodes());
		ImNodes::GetSelectedNodes(selected_nodes.data());
	}

	ImNodes::BeginNodeEditor();

	// Popup Window
	if (ImGui::BeginPopupContextWindow(0, 1, true))
	{
		// Compile
		if (ImGui::MenuItem("Compile"))
		{
			Compile();
			p_device->ClearProfiler();
			recompile = true;
		}

		// Remove
		if (!selected_links.empty() || !selected_nodes.empty())
		{
			if (ImGui::MenuItem("Remove"))
			{
				for (auto &id : selected_links)
				{
					m_edges.erase(static_cast<uint32_t>(id));
				}
				for (auto &id : selected_nodes)
				{
					for (auto iter = m_render_passes.begin(); iter != m_render_passes.end();)
					{
						if ((*iter)->GetHandle() == id)
						{
							for (auto edge_iter = m_edges.begin(); edge_iter != m_edges.end();)
							{
								bool is_erase = false;
								for (auto &resource : (*iter)->GetResources())
								{
									if (edge_iter->second.first == resource || edge_iter->second.second == resource)
									{
										edge_iter = m_edges.erase(edge_iter);
										is_erase  = true;
										break;
									}
								}
								if (!is_erase)
								{
									edge_iter++;
								}
							}
							iter = m_render_passes.erase(iter);
						}
						else
						{
							iter++;
						}
					}
				}
			}
		}

		// Add Pass
		if (ImGui::BeginMenu("New Pass"))
		{
			for (auto &avaliable_pass : s_avaliable_passes)
			{
				if (ImGui::MenuItem(avaliable_pass.c_str()))
				{
					rttr::variant pass_builder = rttr::type::get_by_name(avaliable_pass.c_str()).create();
					rttr::method  meth         = rttr::type::get_by_name(avaliable_pass.c_str()).get_method("Create");
					meth.invoke(pass_builder, *this);
				}
			}
			ImGui::EndMenu();
		}

		// Save Render Graph Config
		if (ImGui::MenuItem("Save"))
		{
			context.OpenFileDialog("Save Render Graph", "Save Render Graph", "Render Graph file (*.rg;){.rg},.*");
		}

		// Save Render Graph Config
		if (ImGui::MenuItem("Load"))
		{
			context.OpenFileDialog("Load Render Graph", "Load Render Graph", "Render Graph file (*.rg;){.rg},.*");
		}

		// Clear All Nodes & Links
		if (ImGui::MenuItem("Clear"))
		{
			m_render_passes.clear();
			m_resources.clear();
			m_edges.clear();
		}

		ImGui::EndPopup();
	}

	// Draw Pass
	for (auto &pass : m_render_passes)
	{
		const float node_width = 200.0f;
		// Inside ImNode context
		ImNodes::BeginNode(pass->GetHandle());
		ImGui::Text(pass->GetName().c_str());

		for (const auto &handle : pass->GetResources())
		{
			auto *resource = m_resources[handle].get();

			// Texture
			if (resource->GetType() == ResourceType::Texture)
			{
				TextureDeclaration *texture = static_cast<TextureDeclaration *>(resource);
				if (IsRead(texture->GetState().access_mask))
				{
					ImNodes::BeginInputAttribute(static_cast<int32_t>(handle));
					ImGui::TextUnformatted(texture->GetName().c_str());
					ImNodes::EndInputAttribute();
				}
				else
				{
					ImNodes::BeginOutputAttribute(static_cast<int32_t>(handle));
					const float label_width = ImGui::CalcTextSize(texture->GetName().c_str()).x;
					ImGui::Indent(node_width - label_width);
					ImGui::TextUnformatted(texture->GetName().c_str());
					ImNodes::EndOutputAttribute();
				}
			}

			// Buffer
			if (resource->GetType() == ResourceType::Buffer)
			{
				BufferDeclaration *buffer = static_cast<BufferDeclaration *>(resource);
				if (IsRead(buffer->GetState().access_mask))
				{
					ImNodes::BeginInputAttribute(static_cast<int32_t>(handle));
					ImGui::TextUnformatted(buffer->GetName().c_str());
					ImNodes::EndInputAttribute();
				}
				else
				{
					ImNodes::BeginOutputAttribute(static_cast<int32_t>(handle));
					const float label_width = ImGui::CalcTextSize(buffer->GetName().c_str()).x;
					ImGui::Indent(node_width - label_width);
					ImGui::TextUnformatted(buffer->GetName().c_str());
					ImNodes::EndOutputAttribute();
				}
			}
		}

		ImNodes::EndNode();
	}

	// Draw Links
	{
		for (auto &[hash, edge] : m_edges)
		{
			ImNodes::Link(static_cast<int32_t>(hash), static_cast<int32_t>(edge.first), static_cast<int32_t>(edge.second));
		}
	}

	ImNodes::MiniMap();
	ImNodes::EndNodeEditor();
	// Create Link
	{
		int32_t from = 0, to = 0;
		if (ImNodes::IsLinkCreated(&from, &to))
		{
			Link(RGHandle(from), RGHandle(to));
		}
	}

	ImGui::End();

	context.GetFileDialogResult("Save Render Graph", [this](const std::string &name) { Save(name); });
	context.GetFileDialogResult("Load Render Graph", [this](const std::string &name) { Load(name); });

	return recompile;
}

void RGBuilder::Save(const std::string &filename)
{
	auto name = Path::GetInstance().GetFileDirectory(filename) + Path::GetInstance().GetFileName(filename, false);

	RGSerializeData data = {};
	for (auto &pass : m_render_passes)
	{
		data.passes.push_back(pass->GetName());
		data.pass_handles.push_back(pass->GetHandle());
		data.nodes.push_back({});
		for (auto &resource : pass->GetResources())
		{
			data.nodes.back().push_back(resource);
		}
	}
	for (auto &[hash, edge] : m_edges)
	{
		data.edges.emplace(edge.first, edge.second);
	}

	std::ofstream             os(name + ".rg", std::ios::binary);
	cereal::JSONOutputArchive archive(os);
	archive(data);

	ImNodes::SaveCurrentEditorStateToIniFile((name + ".ini").c_str());
}

void RGBuilder::Load(const std::string &filename)
{
	m_render_passes.clear();
	m_resources.clear();
	m_edges.clear();

	RGHandle::Reset();

	auto name = Path::GetInstance().GetFileDirectory(filename) + Path::GetInstance().GetFileName(filename, false);

	std::ifstream is(name + ".rg");

	cereal::JSONInputArchive archive(is);
	RGSerializeData          data = {};
	archive(data);

	for (uint32_t i = 0; i < data.passes.size(); i++)
	{
		RGHandle::SetCurrent(data.nodes[i].front());
		rttr::variant pass_builder = rttr::type::get_by_name(data.passes[i].c_str()).create();
		rttr::method  meth         = rttr::type::get_by_name(data.passes[i].c_str()).get_method("Create");
		meth.invoke(pass_builder, *this);
		m_render_passes.back()->SetHandle(data.pass_handles[i]);
	}

	for (auto &[from, to] : data.edges)
	{
		Link(from, to);
	}

	ImNodes::LoadCurrentEditorStateFromIniFile((name + ".ini").c_str());
}

Renderer &RGBuilder::GetRenderer()
{
	return m_renderer;
}

}        // namespace Ilum