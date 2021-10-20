#pragma once

#include "Utils/PCH.hpp"

#include "Editor/Panel.hpp"

#include "Graphics/Image/Image.hpp"

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

	virtual void draw() override;

  private:
	struct Node;

	struct Pin
	{
		ax::NodeEditor::PinId   id;
		std::string             name;
		Node *                  node = nullptr;
		ax::NodeEditor::PinKind kind;

		std::vector<std::pair<std::string, std::string>> infos;

		Pin(int id, const std::string &name, ax::NodeEditor::PinKind kind) :
		    id(id), name(name), kind(kind)
		{
		}
	};

	struct Node
	{
		ax::NodeEditor::NodeId id;
		std::string            name;
		std::vector<Pin>       inputs;
		std::vector<Pin>       outputs;
		ImColor                color;
		ImVec2                 size;

		Node(int id, const std::string& name, ImColor color = ImColor(255, 255, 255)) :
		    id(id), name(name), color(color), size(0, 0)
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

	Pin *findPin(ax::NodeEditor::PinId id);

  private:
	ax::NodeEditor::EditorContext *m_editor_context = nullptr;
	std::vector<Pin>               m_pins;
	std::vector<Node>              m_nodes;
	std::vector<Link>              m_links;

	ax::NodeEditor::PinId m_select_pin;

	ImTextureID m_background_id;
};
}        // namespace Ilum::panel