#include "RenderGraphViewer.hpp"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include <imgui_node_editor_internal.h>

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"

#include "ImGui/ImGuiContext.hpp"

#include <builders.h>
#include <drawing.h>
#include <widgets.h>

namespace ed = ax::NodeEditor;

namespace Ilum::panel
{
RenderGraphViewer::RenderGraphViewer()
{
	ed::Config config;
	config.SettingsFile = "Simple.json";
	m_editor_context    = ed::CreateEditor(&config);

	m_background_id = ImGuiContext::textureID(Renderer::instance()->getResourceCache().loadImage(std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/node_editor_bg.png"), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp));

	Renderer::instance()->Event_RenderGraph_Rebuild += [this]() { build(); };

	ed::SetCurrentEditor(m_editor_context);
}

RenderGraphViewer::~RenderGraphViewer()
{
	ed::DestroyEditor(m_editor_context);
}

void RenderGraphViewer::build()
{
	clear();

	auto &render_graph = Renderer::instance()->getRenderGraph();

	int unique_id = 0;

	for (auto &node : render_graph.getNodes())
	{
		m_nodes.emplace_back(unique_id++, node.name, ImColor(255, 128, 128));

		// Input - Buffer
		for (auto &[set, buffers] : node.descriptors.getBoundBuffers())
		{
			for (auto &buffer : buffers)
			{
				m_nodes.back().inputs.emplace_back(unique_id++, buffer.name, ed::PinKind::Input);
				m_nodes.back().inputs.back().node = &m_nodes.back();
				m_nodes.back().inputs.back().infos.emplace_back("type", "buffer");
				m_nodes.back().inputs.back().infos.emplace_back("name", buffer.name);
				m_nodes.back().inputs.back().infos.emplace_back("set", std::to_string(set));
				m_nodes.back().inputs.back().infos.emplace_back("binding", std::to_string(buffer.binding));
			}
		}

		// Input - image
		for (auto &[set, images] : node.descriptors.getBoundImages())
		{
			for (auto &image : images)
			{
				m_nodes.back().inputs.emplace_back(unique_id++, image.name, ed::PinKind::Input);
				m_nodes.back().inputs.back().node = &m_nodes.back();
				m_nodes.back().inputs.back().infos.emplace_back("type", "image");
				m_nodes.back().inputs.back().infos.emplace_back("name", image.name);
				m_nodes.back().inputs.back().infos.emplace_back("set", std::to_string(set));
				m_nodes.back().inputs.back().infos.emplace_back("binding", std::to_string(image.binding));
			}
		}

		// Output - image
		for (auto &output : node.attachments)
		{
			m_nodes.back().outputs.emplace_back(unique_id++, output, ed::PinKind::Output);
			m_nodes.back().outputs.back().node = &m_nodes.back();
		}
	}

	// Back buffer
	m_nodes.emplace_back(unique_id++, "Back Buffer", ImColor(128, 255, 128));
	m_nodes.back().inputs.emplace_back(unique_id++, render_graph.output(), ed::PinKind::Input);
	m_nodes.back().inputs.back().node = &m_nodes.back();

	// External resource
	m_nodes.emplace_back(unique_id++, "External Resource", ImColor(128, 128, 255));
	m_nodes.back().outputs.emplace_back(unique_id++, "", ed::PinKind::Output);
	m_nodes.back().outputs.back().node = &m_nodes.back();

	// Setting Links
	std::unordered_set<std::string> has_input;
	for (auto &start : m_nodes)
	{
		for (auto &end : m_nodes)
		{
			if (start.id == end.id)
			{
				continue;
			}

			for (auto &output : start.outputs)
			{
				for (auto &input : end.inputs)
				{
					if (input.name == output.name)
					{
						m_links.emplace_back(unique_id++, output.id, input.id);
						has_input.insert(input.name);
					}
				}
			}
		}
	}

	for (auto &node : m_nodes)
	{
		for (auto &input : node.inputs)
		{
			if (has_input.find(input.name) == has_input.end())
			{
				m_links.emplace_back(unique_id++, m_nodes.back().outputs[0].id, input.id);
			}
		}
	}
}

void RenderGraphViewer::clear()
{
	m_links.clear();
	m_nodes.clear();
	m_pins.clear();
}

void RenderGraphViewer::draw()
{
	ImGui::Begin("Render Graph Visualization");
	ed::Begin("Render Graph Visualization", ImVec2(0.0, 0.0f));

	auto &bg           = Renderer::instance()->getResourceCache().loadImage(std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/node_editor_bg.png");
	auto &sampler      = Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp);
	auto &render_graph = Renderer::instance()->getRenderGraph();

	ed::Utilities::BlueprintNodeBuilder builder(ImGuiContext::textureID(bg, sampler), static_cast<int>(bg.get().getWidth()), static_cast<int>(bg.get().getHeight()));

	for (auto &node : m_nodes)
	{
		bool hasOutputDelegates = false;

		builder.Begin(node.id);
		{
			builder.Header(node.color);
			ImGui::Spring(0);
			ImGui::TextUnformatted(node.name.c_str());
			ImGui::Spring(1);
			ImGui::Dummy(ImVec2(0, 28));
			ImGui::Spring(0);
			builder.EndHeader();
		}

		for (auto &input : node.inputs)
		{
			auto alpha = ImGui::GetStyle().Alpha;

			builder.Input(input.id);
			ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
			ax::Widgets::Icon(ImVec2(24, 24), ax::Drawing::IconType::Flow, isPinLink(input.id), ImColor(255, 255, 255), ImColor(32, 32, 32, (int) (alpha * 255)));
			ImGui::Spring(0);
			if (!input.name.empty())
			{
				ImGui::TextUnformatted(input.name.c_str());
				ImGui::Spring(0);
			}
			ImGui::PopStyleVar();
			builder.EndInput();
		}

		for (auto &output : node.outputs)
		{
			auto alpha = ImGui::GetStyle().Alpha;

			ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
			builder.Output(output.id);
			if (!output.name.empty())
			{
				ImGui::Spring(0);
				ImGui::TextUnformatted(output.name.c_str());
			}
			else
			{
				ImGui::BeginVertical(output.id.AsPointer());
				for (auto &image : Renderer::instance()->getResourceCache().getImages())
				{
					ImGui::Image(ImGuiContext::textureID(image, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), {200.f, static_cast<float>(image.getHeight()) * 200.f / static_cast<float>(image.getWidth())});
				}
				ImGui::EndVertical();
			}
			ImGui::Spring(0);
			ax::Widgets::Icon(ImVec2(24, 24), ax::Drawing::IconType::Flow, isPinLink(output.id), ImColor(255, 255, 255), ImColor(32, 32, 32, (int) (alpha * 255)));
			ImGui::PopStyleVar();
			builder.EndOutput();
		}

		builder.End();
	}

	// Linking and flowing
	for (auto &link : m_links)
	{
		ed::Link(link.id, link.start, link.end);
		ed::Flow(link.id);
	}

	// Handle popup
	ed::Suspend();
	if (ed::ShowPinContextMenu(&m_select_pin))
	{
		ImGui::OpenPopup("Pin Context Menu");
	}
	ed::Resume();

	ed::Suspend();
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
	if (ImGui::BeginPopup("Pin Context Menu"))
	{
		auto *pin = findPin(m_select_pin);
		ImGui::TextUnformatted("Pin Context Menu");
		ImGui::Separator();
		if (pin)
		{
			for (auto& [key, info] : pin->infos)
			{
				ImGui::Text("%s: %s", key.c_str(), info.c_str());
			}

			if (pin->name != render_graph.output() && render_graph.hasAttachment(pin->name))
			{
				for (auto& node : m_nodes)
				{
					bool found = false;
					for (auto& intput : node.inputs)
					{
						if (pin->name == intput.name)
						{
							auto &image = render_graph.getAttachment(pin->name);
							ImGui::Image(ImGuiContext::textureID(image, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), {400.f, static_cast<float>(image.getHeight()) * 400.f / static_cast<float>(image.getWidth())});
							found = true;
							break;
						}

						if (found)
						{
							break;
						}
					}
				}
			}
		}
		ImGui::EndPopup();
	}
	ImGui::PopStyleVar();
	ed::Resume();

	ed::End();
	ImGui::End();
}

bool RenderGraphViewer::isPinLink(ax::NodeEditor::PinId id)
{
	if (!id)
		return false;

	for (auto &link : m_links)
		if (link.start == id || link.end == id)
			return true;

	return false;
}

RenderGraphViewer::Pin *RenderGraphViewer::findPin(ax::NodeEditor::PinId id)
{
	for (auto& node : m_nodes)
	{
		for (auto& input : node.inputs)
		{
			if (input.id == id)
			{
				return &input;
			}
		}
		for (auto &output : node.outputs)
		{
			if (output.id == id)
			{
				return &output;
			}
		}
	}

	return nullptr;
}
}        // namespace Ilum::panel