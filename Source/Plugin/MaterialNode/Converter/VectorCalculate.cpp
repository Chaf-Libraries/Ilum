#include "IMaterialNode.hpp"

using namespace Ilum;

class VectorCalculate : public MaterialNode<VectorCalculate>
{
	enum class CalculationType
	{
		Scale,
		Length,
		Distance,

		Dot,
		Cross,

		Addition,
		Substrate,
		Multiplication,
		Division,

		Sin,
		Cos,
		Tan,

		Maximum,
		Minimum,

		Abs,

		Normalize,
	};

	const std::vector<const char *> calculation_types = {
	    "Scale",
	    "Length",
	    "Distance",
	    "Dot",
	    "Cross",
	    "Addition",
	    "Substrate",
	    "Multiplication",
	    "Division",
	    "Sin",
	    "Cos",
	    "Tan",
	    "Maximum",
	    "Minimum",
	    "Abs",
	    "Normalize",
	};

  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("VectorCalculate")
		    .SetCategory("Converter")
		    .SetVariant(CalculationType::Addition)
		    .Input(handle++, "X", MaterialNodePin::Type::Float3, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(0.f))
		    .Input(handle++, "Y", MaterialNodePin::Type::Float3, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(0.f))
		    .Output(handle++, "Out", MaterialNodePin::Type::Float3);
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

CONFIGURATION_MATERIAL_NODE(VectorCalculate)