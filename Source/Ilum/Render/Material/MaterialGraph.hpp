#pragma once

#include "MaterialPinType.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Ilum
{
class MaterialNode;
class AssetManager;
class ImGuiContext;

class MaterialGraph
{
  public:
	MaterialGraph(AssetManager *asset_manager, const std::string &name = "Untitled Material");
	~MaterialGraph();

	const std::string CompileToHLSL();

	void EnableEditor();

	void OnImGui(ImGuiContext &context);

	size_t NewNodeID();
	size_t NewPinID();

	void Link(size_t from, size_t to);
	void UnLink(size_t from, size_t to);
	bool LinkTo(size_t &from, size_t to);

	std::string CallPin(size_t pin);

	void BindPinCallback(size_t pin, std::function<std::string(void)> &&callback);
	void UnbindPinCallback(size_t pin);

	void AddPin(size_t pin, PinType type);
	void SetPin(size_t pin, PinType type);
	void ErasePin(size_t pin);

	void AddNode(std::unique_ptr<MaterialNode> &&node);
	void EraseNode(size_t node);

	AssetManager *GetAssetManager() const;

  private:
	std::string   m_name;
	AssetManager *m_asset_manager = nullptr;

	bool m_enable_editor = false;

	std::vector<std::unique_ptr<MaterialNode>> m_nodes;
	std::unordered_map<size_t, MaterialNode *> m_node_lookup;

	std::map<size_t, std::pair<size_t, size_t>>        m_edges;
	std::map<size_t, std::function<std::string(void)>> m_pin_callbacks;

	std::map<size_t, PinType> m_pin_type;

  private:
	  // ID = 0 for output node
	size_t m_node_id = 1;
	size_t m_pin_id  = 1;
	size_t m_edge_id = 0;
};
}        // namespace Ilum