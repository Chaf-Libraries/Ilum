#include "MaterialNode.hpp"

#include "Render/Material/MaterialGraph.hpp"

namespace Ilum
{
MaterialNode::MaterialNode(const std::string &name, MaterialGraph *material_graph) :
    m_name(name), m_material_graph(material_graph), m_node_id(material_graph->NewNodeID())
{
}

size_t MaterialNode::GetNodeID() const
{
	return m_node_id;
}

const std::string &MaterialNode::GetName() const
{
	return m_name;
}

}        // namespace Ilum