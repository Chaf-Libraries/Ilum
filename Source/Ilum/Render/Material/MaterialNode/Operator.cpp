#include "Operator.hpp"

#include "Render/Material/MaterialGraph.hpp"

#include <RHI/ImGuiContext.hpp>

namespace Ilum::MGNode
{
BinaryOperator::BinaryOperator(const std::string &name, MaterialGraph *material_graph) :
    MaterialNode(name, material_graph),
    m_lhs_pin(material_graph->NewPinID()),
    m_rhs_pin(material_graph->NewPinID()),
    m_output_pin(material_graph->NewPinID())
{
	m_material_graph->AddPin(m_lhs_pin, m_type);
	m_material_graph->AddPin(m_rhs_pin, m_type);
	m_material_graph->AddPin(m_output_pin, m_type);
}

BinaryOperator::~BinaryOperator()
{
	m_material_graph->UnbindPinCallback(m_output_pin);
	m_material_graph->ErasePin(m_lhs_pin);
	m_material_graph->ErasePin(m_rhs_pin);
	m_material_graph->ErasePin(m_output_pin);
}

void BinaryOperator::OnImnode(ImGuiContext &context)
{
	ImNodes::BeginNode(static_cast<int32_t>(m_node_id));

	ImNodes::BeginNodeTitleBar();
	ImGui::Text(m_name.c_str());
	ImNodes::EndNodeTitleBar();

	ImNodes::BeginInputAttribute(static_cast<int32_t>(m_lhs_pin));
	ImGui::Text("A");
	ImNodes::EndInputAttribute();

	ImNodes::BeginOutputAttribute(static_cast<int32_t>(m_output_pin));
	const float label_width = ImGui::CalcTextSize("Out").x;
	const float node_width  = ImGui::CalcTextSize(m_name.c_str()).x;
	ImGui::Indent(node_width - label_width);
	ImGui::Text("Out");
	ImNodes::EndOutputAttribute();

	ImNodes::BeginInputAttribute(static_cast<int32_t>(m_rhs_pin));
	ImGui::Text("B");
	ImNodes::EndInputAttribute();

	ImNodes::EndNode();
}

Addition::Addition(MaterialGraph *material_graph):
    BinaryOperator("Addition", material_graph)
{
	material_graph->BindPinCallback(m_output_pin, [this]() -> std::string {
		size_t lhs = 0, rhs = 0;
		if (m_material_graph->LinkTo(lhs, m_lhs_pin) &&
		    m_material_graph->LinkTo(rhs, m_rhs_pin))
		{
			return fmt::format("({} + {})", m_material_graph->CallPin(lhs), m_material_graph->CallPin(rhs));
		}
		return "";
	});
}

Subtraction::Subtraction(MaterialGraph *material_graph):
    BinaryOperator("Subtraction", material_graph)
{
	material_graph->BindPinCallback(m_output_pin, [this]() -> std::string {
		size_t lhs = 0, rhs = 0;
		if (m_material_graph->LinkTo(lhs, m_lhs_pin) &&
		    m_material_graph->LinkTo(rhs, m_rhs_pin))
		{
			return fmt::format("({} - {})", m_material_graph->CallPin(lhs), m_material_graph->CallPin(rhs));
		}
		return "";
	});
}

Multiplication::Multiplication(MaterialGraph *material_graph):
    BinaryOperator("Multiplication", material_graph)
{
	material_graph->BindPinCallback(m_output_pin, [this]() -> std::string {
		size_t lhs = 0, rhs = 0;
		if (m_material_graph->LinkTo(lhs, m_lhs_pin) &&
		    m_material_graph->LinkTo(rhs, m_rhs_pin))
		{
			return fmt::format("({} * {})", m_material_graph->CallPin(lhs), m_material_graph->CallPin(rhs));
		}
		return "";
	});
}

Division::Division(MaterialGraph *material_graph):
    BinaryOperator("Division", material_graph)
{
	material_graph->BindPinCallback(m_output_pin, [this]() -> std::string {
		size_t lhs = 0, rhs = 0;
		if (m_material_graph->LinkTo(lhs, m_lhs_pin) &&
		    m_material_graph->LinkTo(rhs, m_rhs_pin))
		{
			return fmt::format("({} / {})", m_material_graph->CallPin(lhs), m_material_graph->CallPin(rhs));
		}
		return "";
	});
}

}        // namespace Ilum::MGNode