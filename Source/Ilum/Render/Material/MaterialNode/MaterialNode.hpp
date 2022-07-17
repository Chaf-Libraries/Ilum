#pragma once

#include "Render/Material/MaterialPinType.hpp"

#include <map>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include <imnodes.h>

namespace Ilum
{
class MaterialGraph;
class ImGuiContext;

class MaterialNode
{
  public:
	explicit MaterialNode(const std::string& name, MaterialGraph *material_graph);

	virtual ~MaterialNode() = default;

	virtual void OnImGui(ImGuiContext &context) = 0;
	virtual void OnImnode() = 0;

	size_t GetNodeID() const;

	const std::string &GetName() const;

  protected:
	MaterialGraph *m_material_graph = nullptr;

	std::string m_name = "Untitled Node";

	size_t m_node_id = ~0U;
};

}        // namespace Ilum