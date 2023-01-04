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

	const std::unordered_set<CalculationType> binary_op = {
	    CalculationType::Addition,
	    CalculationType::Substrate,
	    CalculationType::Multiplication,
	    CalculationType::Division,
	    CalculationType::Maximum,
	    CalculationType::Minimum,
	    CalculationType::Dot,
	    CalculationType::Cross,
	    CalculationType::Distance,
	    CalculationType::Scale,
	};

	const std::unordered_set<CalculationType> scale_output = {
	    CalculationType::Length,
	    CalculationType::Distance,
	    CalculationType::Dot,
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
		if (ImGui::Combo("", (int32_t *) (&type), calculation_types.data(), static_cast<int32_t>(calculation_types.size())))
		{
			node_desc.GetPin("Y").enable  = binary_op.find(type) != binary_op.end();
			node_desc.GetPin("Y").type    = type == CalculationType::Scale ? MaterialNodePin::Type::Float : MaterialNodePin::Type::Float3;
			node_desc.GetPin("Y").variant = type == CalculationType::Scale ? Variant(float(0.f)) : Variant(glm::vec3(0.f));
			node_desc.GetPin("Out").type  = scale_output.find(type) != scale_output.end() ? MaterialNodePin::Type::Float : MaterialNodePin::Type::Float3;
		}
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, const MaterialGraphDesc &graph_desc, Renderer *renderer, MaterialCompilationContext *context) override
	{
		if (context->IsCompiled(node_desc))
		{
			return;
		}

		CalculationType type = *node_desc.GetVariant().Convert<CalculationType>();

		std::map<std::string, std::string> parameters;
		context->SetParameter<glm::vec3>(parameters, node_desc.GetPin("X"), graph_desc, renderer, context);
		if (type == CalculationType::Scale)
		{
			context->SetParameter<float>(parameters, node_desc.GetPin("Y"), graph_desc, renderer, context);
		}
		else
		{
			context->SetParameter<glm::vec3>(parameters, node_desc.GetPin("Y"), graph_desc, renderer, context);
		}

		switch (type)
		{
			case CalculationType::Scale:
				context->variables.emplace_back(fmt::format("float3 S_{} = {} * {};", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Length:
				context->variables.emplace_back(fmt::format("float S_{} = length({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Distance:
				context->variables.emplace_back(fmt::format("float S_{} = length({} - {});", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Dot:
				context->variables.emplace_back(fmt::format("float S_{} = dot({}, {});", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Cross:
				context->variables.emplace_back(fmt::format("float S_{} = cross({}, {});", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Addition:
				context->variables.emplace_back(fmt::format("float3 S_{} = {} + {};", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Substrate:
				context->variables.emplace_back(fmt::format("float3 S_{} = {} - {};", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Multiplication:
				context->variables.emplace_back(fmt::format("float3 S_{} = {} * {};", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Division:
				context->variables.emplace_back(fmt::format("float3 S_{} = {} / {};", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Sin:
				context->variables.emplace_back(fmt::format("float3 S_{} = sin({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Cos:
				context->variables.emplace_back(fmt::format("float3 S_{} = cos({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Tan:
				context->variables.emplace_back(fmt::format("float3 S_{} = tan({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Maximum:
				context->variables.emplace_back(fmt::format("float3 S_{} = max({}, {});", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Minimum:
				context->variables.emplace_back(fmt::format("float3 S_{} = min({}, {});", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Abs:
				context->variables.emplace_back(fmt::format("float3 S_{} = abs({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Normalize:
				context->variables.emplace_back(fmt::format("float S_{} = normalize({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			default:
				break;
		}
	}
};

CONFIGURATION_MATERIAL_NODE(VectorCalculate)