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
	m_name = "Render Graph Viewer";

	ed::Config config;
	config.SettingsFile = "Simple.json";
	m_editor_context    = ed::CreateEditor(&config);

	m_background_id = ImGuiContext::textureID(Renderer::instance()->getResourceCache().loadImage(std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/node_editor_bg.png"), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp));

	Renderer::instance()->Event_RenderGraph_Rebuild += [this]() { build(); };

	ed::SetCurrentEditor(m_editor_context);

	build();
}

RenderGraphViewer::~RenderGraphViewer()
{
	ed::DestroyEditor(m_editor_context);
}

void RenderGraphViewer::build()
{
	clear();

	auto *render_graph = Renderer::instance()->getRenderGraph();

	int unique_id = 0;

	// Pass Node
	for (auto &node : render_graph->getNodes())
	{
		m_passes.emplace_back(unique_id++, node.name, ImColor(255, 128, 128));

		// Descriptor binding
		for (auto &[set, buffers] : node.descriptors.getBoundBuffers())
		{
			for (auto &buffer : buffers)
			{
				std::vector<std::string> buffer_infos;
				buffer_infos.push_back("type: buffer\n");
				buffer_infos.push_back("name: " + buffer.name + "\n");
				buffer_infos.push_back("set: " + std::to_string(set) + "\n");
				buffer_infos.push_back("bind: " + std::to_string(buffer.binding));
				m_passes.back().infos.emplace_back(buffer_infos);
			}
		}

		// Input - image
		for (auto &[set, images] : node.descriptors.getBoundImages())
		{
			for (auto &image : images)
			{
				std::vector<std::string> image_infos;
				image_infos.push_back("type: image\n");
				image_infos.push_back("name: " + image.name + "\n");
				image_infos.push_back("set: " + std::to_string(set) + "\n");
				image_infos.push_back("bind: " + std::to_string(image.binding));
				m_passes.back().infos.emplace_back(image_infos);

				if (std::find_if(render_graph->getAttachments().begin(), render_graph->getAttachments().end(), [&image](const std::pair<const std::string, Image> &iter) { return iter.first == image.name; }) != render_graph->getAttachments().end())
				{
					m_passes.back().inputs.emplace_back(unique_id++, image.name, ed::PinKind::Input);
					m_passes.back().inputs.back().node = &m_passes.back();
				}
			}
		}

		// Output - image
		for (auto &output : node.attachments)
		{
			m_passes.back().outputs.emplace_back(unique_id++, output, ed::PinKind::Output);
			m_passes.back().outputs.back().node = &m_passes.back();
		}
	}

	// Attachment node
	for (auto &[name, image] : render_graph->getAttachments())
	{
		m_attachments.emplace_back(unique_id++, Pin(unique_id++, "", ed::PinKind::Input), name, ImColor(128, 128, 255));
		for (auto &pass : m_passes)
		{
			for (auto &input : pass.inputs)
			{
				if (input.name == name)
				{
					m_attachments.back().output = Pin(unique_id++, "", ed::PinKind::Output);
				}
			}
		}

		if (!m_attachments.back().output)
		{
			m_attachments.back().color = ImColor(128, 255, 128);
		}
	}

	// Setting Links
	for (auto &pass : m_passes)
	{
		for (auto &attachment : m_attachments)
		{
			for (auto &input : pass.inputs)
			{
				if (input.name == attachment.name)
				{
					m_links.emplace_back(unique_id++, attachment.output.value().id, input.id);
				}
			}
			for (auto &output : pass.outputs)
			{
				if (output.name == attachment.name)
				{
					m_links.emplace_back(unique_id++, output.id, attachment.input.id);
				}
			}
		}
	}
}

void RenderGraphViewer::clear()
{
	m_links.clear();
	m_passes.clear();
	m_pins.clear();
	m_attachments.clear();
}

void RenderGraphViewer::draw()
{
	ImGui::Begin("Render Graph Visualization", &active);
	ed::Begin("Render Graph Visualization", ImVec2(0.0, 0.0f));

	auto &bg           = Renderer::instance()->getResourceCache().loadImage(std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/node_editor_bg.png");
	auto &sampler      = Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp);
	auto *render_graph = Renderer::instance()->getRenderGraph();

	ed::Utilities::BlueprintNodeBuilder builder(ImGuiContext::textureID(bg, sampler), static_cast<int>(bg.get().getWidth()), static_cast<int>(bg.get().getHeight()));

	// Draw render pass
	for (auto &node : m_passes)
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
			ImGui::Spring(0);
			ax::Widgets::Icon(ImVec2(24, 24), ax::Drawing::IconType::Flow, isPinLink(output.id), ImColor(255, 255, 255), ImColor(32, 32, 32, (int) (alpha * 255)));
			ImGui::PopStyleVar();
			builder.EndOutput();
		}

		builder.End();
	}

	// Draw attachment
	for (auto &attachment : m_attachments)
	{
		bool hasOutputDelegates = false;

		builder.Begin(attachment.id);
		{
			builder.Header(attachment.color);
			ImGui::Spring(0);
			ImGui::TextUnformatted(attachment.name.c_str());
			ImGui::Spring(1);
			ImGui::Dummy(ImVec2(0, 28));
			ImGui::Spring(0);
			builder.EndHeader();
		}

		//for (auto &input : node.inputs)
		{
			auto alpha = ImGui::GetStyle().Alpha;

			builder.Input(attachment.input.id);
			ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
			ax::Widgets::Icon(ImVec2(12, 12), ax::Drawing::IconType::Flow, isPinLink(attachment.input.id), ImColor(255, 255, 255), ImColor(32, 32, 32, (int) (alpha * 255)));
			ImGui::Spring(0);
			if (Renderer::instance()->isDebug())
			{
				auto &image = render_graph->getAttachment(attachment.name);
				ImGui::Image(ImGuiContext::textureID(image, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), {200.f, static_cast<float>(image.getHeight()) * 200.f / static_cast<float>(image.getWidth())});
			}
			ImGui::PopStyleVar();
			builder.EndInput();
		}

		if (attachment.output)
		{
			auto alpha = ImGui::GetStyle().Alpha;

			ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
			builder.Output(attachment.output.value().id);
			//ImGui::Spring(0);
			ax::Widgets::Icon(ImVec2(12, 12), ax::Drawing::IconType::Flow, isPinLink(attachment.output.value().id), ImColor(255, 255, 255), ImColor(32, 32, 32, (int) (alpha * 255)));
			ImGui::PopStyleVar();
			builder.EndOutput();
		}

		builder.End();
	}

	/*for (auto& attachment : m_attachments)
	{
		ed::BeginNode(attachment.id);
		ImGui::Text(attachment.name.c_str());
		ed::BeginPin(attachment.input.id, ed::PinKind::Input);
		ImGui::Text("-> In");
		ed::EndPin();

		if (Renderer::instance()->isDebug())
		{
			auto &image = render_graph->getAttachment(attachment.name);
			ImGui::Image(ImGuiContext::textureID(image, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), {200.f, static_cast<float>(image.getHeight()) * 200.f / static_cast<float>(image.getWidth())});
		}

		ImGui::SameLine();
		if (attachment.output)
		{
			ed::BeginPin(attachment.output.value().id, ed::PinKind::Output);
			ImGui::Text("Out ->");

			ed::EndPin();
		}

		ed::EndNode();
	}*/

	// Linking and flowing
	for (auto &link : m_links)
	{
		ed::Link(link.id, link.start, link.end);
		ed::Flow(link.id);
	}

	// Handle popup
	//ed::Suspend();
	//if (ed::ShowPinContextMenu(&m_select_pin))
	//{
	//	ImGui::OpenPopup("Pin Context Menu");
	//}
	//ed::Resume();

	//ed::Suspend();
	//ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
	//if (ImGui::BeginPopup("Pin Context Menu"))
	//{
	//	auto *pin = findPin(m_select_pin);
	//	ImGui::TextUnformatted("Pin Context Menu");
	//	ImGui::Separator();
	//	if (pin)
	//	{
	//		//for (auto &[key, info] : pin->infos)
	//		{
	//			//ImGui::Text("%s: %s", key.c_str(), info.c_str());
	//		}

	//		for (auto &[name, output] : render_graph->getAttachments())
	//		{
	//			//if (name != render_graph->output())
	//			//ImGui::Image(ImGuiContext::textureID(output, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), {400.f, static_cast<float>(output.getHeight()) * 400.f / static_cast<float>(output.getWidth())});
	//		}

	//		//if (pin->name != render_graph->output() && render_graph->hasAttachment(pin->name))
	//		//{
	//		//	for (auto& node : m_nodes)
	//		//	{
	//		//		bool found = false;
	//		//		for (auto& intput : node.outputs)
	//		//		{
	//		//			//if (pin->name == intput.name)
	//		//			{
	//		//				auto &image = render_graph->getAttachment(pin->name);
	//		//				ImGui::Image(ImGuiContext::textureID(image, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), {400.f, static_cast<float>(image.getHeight()) * 400.f / static_cast<float>(image.getWidth())});
	//		//				found = true;
	//		//				//break;
	//		//			}

	//		//			if (found)
	//		//			{
	//		//				//break;
	//		//			}
	//		//		}
	//		//	}
	//		//}
	//	}
	//	ImGui::EndPopup();
	//}
	//ImGui::PopStyleVar();
	//ed::Resume();

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
	for (auto &node : m_passes)
	{
		for (auto &input : node.inputs)
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