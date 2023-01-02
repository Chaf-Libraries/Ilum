#include "IMaterialNode.hpp"

using namespace Ilum;

class Calculate : public MaterialNode<Calculate>
{
	enum class CalculationType
	{
		Addition,
		Substrate,
		Multiplication,
		Division,

		Maximum,
		Minimum,

		Greater,
		Less,

		Square,
		Log,
		Exp,
		Sqrt,
		Rcp,
		Abs,
		Sign,

		Sin,
		Cos,
		Tan,

		Asin,
		Acos,
		Atan,
		Atan2,

		Sinh,
		Cosh,
		Tanh,
	};

	const std::vector<const char *> calculation_types = {
	    "Addition",
	    "Substrate",
	    "Multiplication",
	    "Division",
	    "Maximum",
	    "Minimum",
	    "Greater",
	    "Less",
	    "Square",
	    "Log",
	    "Exp",
	    "Sqrt",
	    "Rcp",
	    "Abs",
	    "Sign",
	    "Sin",
	    "Cos",
	    "Tan",
	    "Asin",
	    "Acos",
	    "Atan",
	    "Atan2",
	    "Sinh",
	    "Cosh",
	    "Tanh",
	};

  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("Calculate")
		    .SetCategory("Converter")
		    .SetVariant(CalculationType::Addition)
		    .Input(handle++, "X", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Input(handle++, "Y", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Output(handle++, "Out", MaterialNodePin::Type::Float);
	}

	virtual void OnImGui(MaterialNodeDesc &node_desc, Editor *editor) override
	{
		CalculationType &type = *node_desc.GetVariant().Convert<CalculationType>();
		ImGui::Combo("", (int32_t *) (&type), calculation_types.data(), static_cast<int32_t>(calculation_types.size()));
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, MaterialGraph *graph, MaterialCompilationContext &context) override
	{
	}
};

CONFIGURATION_MATERIAL_NODE(Calculate)