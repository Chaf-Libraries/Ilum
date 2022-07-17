#pragma once

#include "MaterialNode.hpp"

#include "Render/Material/MaterialGraph.hpp"

#include <Core/Log.hpp>

#include <imgui.h>
#include <imnodes.h>

#include <glm/detail/type_half.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <typeindex>

struct half
{
	float v;
};

namespace Ilum::MGNode
{
inline const std::unordered_map<std::type_index, std::array<PinType, 4>> ConstantPinType = {
    {typeid(float), {PinType::Float, PinType::Float2, PinType::Float3, PinType::Float4}},
    {typeid(half), {PinType::Half, PinType::Half2, PinType::Half3, PinType::Half4}},
    {typeid(double), {PinType::Double, PinType::Double2, PinType::Double3, PinType::Double4}},
    {typeid(int32_t), {PinType::Int, PinType::Int2, PinType::Int3, PinType::Int4}},
    {typeid(uint32_t), {PinType::Uint, PinType::Uint2, PinType::Uint3, PinType::Uint4}},
    {typeid(bool), {PinType::Bool, PinType::Bool2, PinType::Bool3, PinType::Bool4}}};

inline const std::unordered_map<std::type_index, std::array<std::string, 4>> ConstantPinTypeName = {
    {typeid(float), {"Float", "Float2", "Float3", "Float4"}},
    {typeid(half), {"Half", "Half2", "Half3", "Half4"}},
    {typeid(double), {"Double", "Double2", "Double3", "Double4"}},
    {typeid(int32_t), {"Int", "Int2", "Int3", "Int4"}},
    {typeid(uint32_t), {"Uint", "Uint2", "Uint3", "Uint4"}},
    {typeid(bool), {"Bool", "Bool2", "Bool3", "Bool4"}}};

template <typename T>
inline std::string to_string(const glm::vec<1, T> &v)
{
	return fmt::format("{}1({})", typeid(T).name(), v.x);
}

template <typename T>
inline std::string to_string(const glm::vec<2, T> &v)
{
	return fmt::format("{}2({}, {})", typeid(T).name(), v.x, v.y);
}

template <typename T>
inline std::string to_string(const glm::vec<3, T> &v)
{
	return fmt::format("{}3({}, {}, {})", typeid(T).name(), v.x, v.y, v.z);
}

template <typename T>
inline std::string to_string(const glm::vec<4, T> &v)
{
	return fmt::format("{}4({}, {}, {}, {})", typeid(T).name(), v.x, v.y, v.z, v.w);
}

template <>
inline std::string to_string(const glm::vec<1, half> &v)
{
	return fmt::format("half({})", v.x.v);
}

template <>
inline std::string to_string(const glm::vec<2, half> &v)
{
	return fmt::format("half2({}, {})", v.x.v, v.y.v);
}

template <>
inline std::string to_string(const glm::vec<3, half> &v)
{
	return fmt::format("half3({}, {}, {})", v.x.v, v.y.v, v.z.v);
}

template <>
inline std::string to_string(const glm::vec<4, half> &v)
{
	return fmt::format("half4({}, {}, {}, {})", v.x.v, v.y.v, v.z.v, v.w.v);
}

template <typename T, int32_t L>
struct EditConstant
{
	void operator()(glm::vec<L, T> &v)
	{
	}
};

template <int32_t L>
struct EditConstant<float, L>
{
	void operator()(glm::vec<L, float> &v)
	{
		ImGui::DragScalarN("Data", ImGuiDataType_Float, &v.x, L, 0.01f);
	}
};

template <int32_t L>
struct EditConstant<int32_t, L>
{
	void operator()(glm::vec<L, int32_t> &v)
	{
		ImGui::DragScalarN("Data", ImGuiDataType_S32, &v.x, L, 0.01f);
	}
};

template <int32_t L>
struct EditConstant<uint32_t, L>
{
	void operator()(glm::vec<L, uint32_t> &v)
	{
		ImGui::DragScalarN("Data", ImGuiDataType_U32, &v.x, L, 0.01f);
	}
};

template <int32_t L>
struct EditConstant<bool, L>
{
	void operator()(glm::vec<L, bool> &v)
	{
		for (int32_t i = 0; i < L; i++)
		{
			ImGui::Checkbox("", &v[i]);
			ImGui::SameLine();
		}
		ImGui::Text("Data");
	}
};

template <int32_t L>
struct EditConstant<half, L>
{
	void operator()(glm::vec<L, half> &v)
	{
		ImGui::DragScalarN("Data", ImGuiDataType_Float, &v.x, L, 0.01f);
	}
};

template <int32_t L>
struct EditConstant<double, L>
{
	void operator()(glm::vec<L, double> &v)
	{
		ImGui::DragScalarN("Data", ImGuiDataType_Double, &v.x, L, 0.01f);
	}
};

template <typename T, int32_t L>
class Constant : public MaterialNode
{
  public:
	Constant(MaterialGraph *material_graph) :
	    MaterialNode(fmt::format("Constant {}", ConstantPinTypeName.at(typeid(T))[L - 1]), material_graph),
	    m_data_pin(material_graph->NewPinID())
	{
		material_graph->BindPinCallback(m_data_pin, [this]() -> std::string {
			return to_string(m_data);
		});

		material_graph->AddPin(m_data_pin, ConstantPinType.at(typeid(T))[L - 1]);
	}

	virtual ~Constant()
	{
		m_material_graph->UnbindPinCallback(m_data_pin);
		m_material_graph->ErasePin(m_data_pin);
	}

	virtual void OnImGui(ImGuiContext &context) override
	{
		EditConstant<T, L> edit;
		edit(m_data);
	}

	virtual void OnImnode() override
	{
		ImNodes::BeginNode(static_cast<int32_t>(m_node_id));

		ImNodes::BeginNodeTitleBar();
		ImGui::Text(m_name.c_str());
		ImNodes::EndNodeTitleBar();

		ImNodes::BeginOutputAttribute(static_cast<int32_t>(m_data_pin));
		const float label_width = ImGui::CalcTextSize("Out").x;
		const float node_width  = ImGui::CalcTextSize(m_name.c_str()).x;
		ImGui::Indent(node_width - label_width);
		ImGui::Text("Out");
		ImNodes::EndOutputAttribute();

		ImNodes::EndNode();
	}

  private:
	size_t m_data_pin = ~0U;

	glm::vec<L, T> m_data = {};
};

using Float  = Constant<float, 1>;
using Half   = Constant<half, 1>;
using Double = Constant<double, 1>;
using Int    = Constant<int32_t, 1>;
using Uint   = Constant<uint32_t, 1>;
using Bool   = Constant<bool, 1>;

using Float2  = Constant<float, 2>;
using Half2   = Constant<half, 2>;
using Double2 = Constant<double, 2>;
using Int2    = Constant<int32_t, 2>;
using Uint2   = Constant<uint32_t, 2>;
using Bool2   = Constant<bool, 2>;

using Float3  = Constant<float, 3>;
using Half3   = Constant<half, 3>;
using Double3 = Constant<double, 3>;
using Int3    = Constant<int32_t, 3>;
using Uint3   = Constant<uint32_t, 3>;
using Bool3   = Constant<bool, 3>;

using Float4  = Constant<float, 4>;
using Half4   = Constant<half, 4>;
using Double4 = Constant<double, 4>;
using Int4    = Constant<int32_t, 4>;
using Uint4   = Constant<uint32_t, 4>;
using Bool4   = Constant<bool, 4>;

inline static std::map<const char *, std::map<const char *, std::function<std::unique_ptr<MaterialNode>(MaterialGraph *)>>> ConstantNodeCreation = {
#define CONSTANT_NODE_CREATION_FUNC(Type)                                                            \
	{                                                                                                \
#		Type, [](MaterialGraph * material_graph) { return std::make_unique<Type>(material_graph); } \
	}

    {"Scalar",
     {CONSTANT_NODE_CREATION_FUNC(Float),
      CONSTANT_NODE_CREATION_FUNC(Half),
      CONSTANT_NODE_CREATION_FUNC(Double),
      CONSTANT_NODE_CREATION_FUNC(Int),
      CONSTANT_NODE_CREATION_FUNC(Uint),
      CONSTANT_NODE_CREATION_FUNC(Bool)}},

    {"Vec2",
     {CONSTANT_NODE_CREATION_FUNC(Float2),
      CONSTANT_NODE_CREATION_FUNC(Half2),
      CONSTANT_NODE_CREATION_FUNC(Double2),
      CONSTANT_NODE_CREATION_FUNC(Int2),
      CONSTANT_NODE_CREATION_FUNC(Uint2),
      CONSTANT_NODE_CREATION_FUNC(Bool2)}},

    {"Vec3",
     {CONSTANT_NODE_CREATION_FUNC(Float3),
      CONSTANT_NODE_CREATION_FUNC(Half3),
      CONSTANT_NODE_CREATION_FUNC(Double3),
      CONSTANT_NODE_CREATION_FUNC(Int3),
      CONSTANT_NODE_CREATION_FUNC(Uint3),
      CONSTANT_NODE_CREATION_FUNC(Bool3)}},

    {"Vec4",
     {CONSTANT_NODE_CREATION_FUNC(Float4),
      CONSTANT_NODE_CREATION_FUNC(Half4),
      CONSTANT_NODE_CREATION_FUNC(Double4),
      CONSTANT_NODE_CREATION_FUNC(Int4),
      CONSTANT_NODE_CREATION_FUNC(Uint4),
      CONSTANT_NODE_CREATION_FUNC(Bool4)}},
};
}        // namespace Ilum::MGNode