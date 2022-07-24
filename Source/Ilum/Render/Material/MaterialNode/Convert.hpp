#pragma once

#include "MaterialNode.hpp"

#include "RHI/ImGuiContext.hpp"

#include <functional>

namespace Ilum::MGNode
{
template <size_t Size>
class SplitVector: public MaterialNode
{
  public:
	SplitVector(MaterialGraph *material_graph) :
	    MaterialNode(fmt::format("Vec{} Split", Size), material_graph),
	    m_in_pin(material_graph->NewPinID())
	{
		const char *const components[] = {"x", "y", "z", "w"};

		material_graph->AddPin(m_in_pin, (PinType) (1 << (Size - 1)));

		for (size_t i = 0; i < Size; i++)
		{
			size_t new_pin = material_graph->NewPinID();
			m_out_pins[i]  = new_pin;
			material_graph->AddPin(new_pin, PinType::Scalar);
			material_graph->BindPinCallback(new_pin, [this, components, i]() -> std::string {
				size_t from = 0;
				if (m_material_graph->LinkTo(from, m_in_pin))
				{
					return fmt::format("({}).{}", m_material_graph->CallPin(from), components[i]);
				}
				return "";
			});
		}
	}

	virtual ~SplitVector() override
	{
		for (size_t i = 0; i < Size; i++)
		{
			m_material_graph->UnbindPinCallback(m_out_pins[i]);
			m_material_graph->ErasePin(m_out_pins[i]);
		}
	}

	virtual void OnImnode(ImGuiContext &context) override
	{
		ImNodes::BeginNode(static_cast<int32_t>(m_node_id));

		ImNodes::BeginNodeTitleBar();
		ImGui::Text(m_name.c_str());
		ImNodes::EndNodeTitleBar();

		ImNodes::BeginInputAttribute(static_cast<int32_t>(m_in_pin));
		ImGui::Text("In");
		ImNodes::EndInputAttribute();

		const char *const components[] = {"x", "y", "z", "w"};

		for (size_t i = 0; i < Size; i++)
		{
			ImNodes::BeginOutputAttribute(static_cast<int32_t>(m_out_pins[i]));
			const float label_width = ImGui::CalcTextSize(components[i]).x;
			const float node_width  = ImGui::CalcTextSize(m_name.c_str()).x;
			ImGui::Indent(node_width - label_width);
			ImGui::Text(components[i]);
			ImNodes::EndOutputAttribute();
		}

		ImNodes::EndNode();
	}

  private:
	std::array<size_t, Size> m_out_pins;
	size_t                   m_in_pin;
};

inline static std::map<const char *, std::function<std::unique_ptr<MaterialNode>(MaterialGraph *)>> ConvertNodeCreation = {
    {"Split Vec2", [](MaterialGraph *material_graph) { return std::make_unique<SplitVector<2>>(material_graph); }},
    {"Split Vec3", [](MaterialGraph *material_graph) { return std::make_unique<SplitVector<3>>(material_graph); }},
    {"Split Vec4", [](MaterialGraph *material_graph) { return std::make_unique<SplitVector<4>>(material_graph); }}
};
}        // namespace Ilum::MGNode