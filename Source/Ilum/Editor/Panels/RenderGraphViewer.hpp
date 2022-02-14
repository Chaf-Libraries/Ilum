#pragma once

#include "Utils/PCH.hpp"

#include "Editor/Panel.hpp"

#include <Graphics/Resource/Image.hpp>

#include <imgui_node_editor.h>

namespace Ilum::panel
{
class RenderGraphViewer : public Panel
{
  public:
	RenderGraphViewer();

	~RenderGraphViewer();

	void build();

	void clear();

	virtual void draw(float delta_time) override;

  private:
	struct PassNode;

	struct Pin
	{
		ax::NodeEditor::PinId   id;
		std::string             name;
		PassNode *              node = nullptr;
		ax::NodeEditor::PinKind kind;

		Pin(int id, const std::string &name, ax::NodeEditor::PinKind kind) :
		    id(id), name(name), kind(kind)
		{
		}
	};

	struct PassNode
	{
		ax::NodeEditor::NodeId                id;
		std::string                           name;
		std::vector<Pin>                      inputs;
		std::vector<Pin>                      outputs;
		std::vector<std::vector<std::string>> infos;
		ImColor                               color;
		ImVec2                                size;

		PassNode(int id, const std::string &name, ImColor color = ImColor(255, 255, 255)) :
		    id(id), name(name), color(color), size(0, 0)
		{
		}
	};

	struct AttachmentNode
	{
		ax::NodeEditor::NodeId id;
		std::string            name;
		Pin                    input;
		std::optional<Pin>     output;
		ImColor                color;
		ImVec2                 size;

		AttachmentNode(int id, Pin& input, const std::string &name, ImColor color = ImColor(255, 255, 255)) :
		    id(id), input(input), name(name), color(color), size(0, 0)
		{
		}
	};

	struct Link
	{
		ax::NodeEditor::LinkId id;

		ax::NodeEditor::PinId start;
		ax::NodeEditor::PinId end;

		ImColor color;

		Link(ax::NodeEditor::LinkId id, ax::NodeEditor::PinId start, ax::NodeEditor::PinId end) :
		    id(id), start(start), end(end), color(255, 255, 255)
		{
		}
	};

  private:
	bool isPinLink(ax::NodeEditor::PinId id);

	PassNode *findPassNode(ax::NodeEditor::NodeId id);

	AttachmentNode *findAttachmentNode(ax::NodeEditor::NodeId id);

  private:
	ax::NodeEditor::EditorContext *m_editor_context = nullptr;
	std::vector<Pin>               m_pins;
	std::vector<PassNode>          m_passes;
	std::vector<AttachmentNode>    m_attachments;
	std::vector<Link>              m_links;

	ax::NodeEditor::NodeId m_select_node;

	Graphics::Image m_bg;
};
}        // namespace Ilum::panel