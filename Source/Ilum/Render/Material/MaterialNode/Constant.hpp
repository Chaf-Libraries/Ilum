#pragma once

#include "MaterialNode.hpp"

#include "Render/Material/MaterialGraph.hpp"

#include <Core/Log.hpp>

#include <imgui.h>
#include <imnodes.h>

#include <glm/detail/type_half.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <typeindex>

namespace Ilum::MGNode
{
inline std::string to_string(const glm::vec<1, float> &v)
{
	return fmt::format("float({})", v.x);
}

inline std::string to_string(const glm::vec<2, float> &v)
{
	return fmt::format("float2({}, {})", v.x, v.y);
}

inline std::string to_string(const glm::vec<3, float> &v)
{
	return fmt::format("float3({}, {}, {})", v.x, v.y, v.z);
}

inline std::string to_string(const glm::vec<4, float> &v)
{
	return fmt::format("float4({}, {}, {}, {})", v.x, v.y, v.z, v.w);
}

template <size_t Size>
class Constant : public MaterialNode
{
  public:
	Constant(MaterialGraph *material_graph) :
	    MaterialNode(Size == 1 ? "Scalar" : fmt::format("Vec{}", Size), material_graph),
	    m_data_pin(material_graph->NewPinID())
	{
		material_graph->BindPinCallback(m_data_pin, [this]() -> std::string {
			return to_string(m_data);
		});

		material_graph->AddPin(m_data_pin, static_cast<PinType>(1 << (Size - 1)));
	}

	virtual ~Constant()
	{
		m_material_graph->UnbindPinCallback(m_data_pin);
		m_material_graph->ErasePin(m_data_pin);
	}

	virtual void OnImnode(ImGuiContext &context) override
	{
		ImNodes::BeginNode(static_cast<int32_t>(m_node_id));

		ImNodes::BeginNodeTitleBar();
		ImGui::Text(m_name.c_str());
		ImNodes::EndNodeTitleBar();

		const float label_width = ImGui::CalcTextSize("Out").x;
		float node_width  = ImGui::CalcTextSize(m_name.c_str()).x;

		node_width *= Size * 2;

		ImGui::PushItemWidth(node_width);
		ImGui::DragScalarN("", ImGuiDataType_Float, &m_data.x, Size, 0.1f);
		ImGui::PopItemWidth();

		ImNodes::BeginOutputAttribute(static_cast<int32_t>(m_data_pin));

		ImGui::Indent(node_width - label_width);
		ImGui::Text("Out");
		ImNodes::EndOutputAttribute();

		ImNodes::EndNode();
	}

  private:
	size_t m_data_pin = ~0U;

	glm::vec<Size, float> m_data = {};
};

inline static std::map<const char *, std::function<std::unique_ptr<MaterialNode>(MaterialGraph *)>> ConstantNodeCreation = {
    {"Scalar", [](MaterialGraph *material_graph) { return std::make_unique<Constant<1>>(material_graph); }},
    {"Vec2", [](MaterialGraph *material_graph) { return std::make_unique<Constant<2>>(material_graph); }},
    {"Vec3", [](MaterialGraph *material_graph) { return std::make_unique<Constant<3>>(material_graph); }},
    {"Vec4", [](MaterialGraph *material_graph) { return std::make_unique<Constant<4>>(material_graph); }}};
}        // namespace Ilum::MGNode