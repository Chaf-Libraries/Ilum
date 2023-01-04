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

	const std::unordered_set<CalculationType> binary_op = {
	    CalculationType::Addition,
	    CalculationType::Substrate,
	    CalculationType::Multiplication,
	    CalculationType::Division,
	    CalculationType::Maximum,
	    CalculationType::Minimum,
	    CalculationType::Greater,
	    CalculationType::Less,
	    CalculationType::Square,
	    CalculationType::Atan2,
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
		if (ImGui::Combo("", (int32_t *) (&type), calculation_types.data(), static_cast<int32_t>(calculation_types.size())))
		{
			node_desc.GetPin("Y").enable = binary_op.find(type) != binary_op.end();
		}
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, const MaterialGraphDesc &graph_desc, Renderer *renderer, MaterialCompilationContext *context) override
	{
		if (context->IsCompiled(node_desc))
		{
			return;
		}

		std::map<std::string, std::string> parameters;
		context->SetParameter<float>(parameters, node_desc.GetPin("X"), graph_desc, renderer, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Y"), graph_desc, renderer, context);

		CalculationType type = *node_desc.GetVariant().Convert<CalculationType>();

		switch (type)
		{
			case CalculationType::Addition:
				context->variables.emplace_back(fmt::format("float S_{} = {} + {};", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Substrate:
				context->variables.emplace_back(fmt::format("float S_{} = {} - {};", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Multiplication:
				context->variables.emplace_back(fmt::format("float S_{} = {} * {};", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Division:
				context->variables.emplace_back(fmt::format("float S_{} = {} / {};", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Maximum:
				context->variables.emplace_back(fmt::format("float S_{} = max({}, {});", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Minimum:
				context->variables.emplace_back(fmt::format("float S_{} = min({}, {});", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Greater:
				context->variables.emplace_back(fmt::format("float S_{} = {} > {};", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Less:
				context->variables.emplace_back(fmt::format("float S_{} = {} < {};", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"]));
				break;
			case CalculationType::Square:
				context->variables.emplace_back(fmt::format("float S_{} = {} * {};", node_desc.GetPin("Out").handle, parameters["X"], parameters["X"]));
				break;
			case CalculationType::Log:
				context->variables.emplace_back(fmt::format("float S_{} = log({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Exp:
				context->variables.emplace_back(fmt::format("float S_{} = exp({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Sqrt:
				context->variables.emplace_back(fmt::format("float S_{} = sqrt({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Rcp:
				context->variables.emplace_back(fmt::format("float S_{} = rcp({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Abs:
				context->variables.emplace_back(fmt::format("float S_{} = abs({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Sign:
				context->variables.emplace_back(fmt::format("float S_{} = sign({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Sin:
				context->variables.emplace_back(fmt::format("float S_{} = sin({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Cos:
				context->variables.emplace_back(fmt::format("float S_{} = cos({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Tan:
				context->variables.emplace_back(fmt::format("float S_{} = tan({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Asin:
				context->variables.emplace_back(fmt::format("float S_{} = asin({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Acos:
				context->variables.emplace_back(fmt::format("float S_{} = acos({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Atan:
				context->variables.emplace_back(fmt::format("float S_{} = atan({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Atan2:
				context->variables.emplace_back(fmt::format("float S_{} = atan2({}, {});", node_desc.GetPin("Out").handle, parameters["Y"], parameters["X"]));
				break;
			case CalculationType::Sinh:
				context->variables.emplace_back(fmt::format("float S_{} = sinh({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Cosh:
				context->variables.emplace_back(fmt::format("float S_{} = cosh({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			case CalculationType::Tanh:
				context->variables.emplace_back(fmt::format("float S_{} = tanh({});", node_desc.GetPin("Out").handle, parameters["X"]));
				break;
			default:
				break;
		}
	}
};

CONFIGURATION_MATERIAL_NODE(Calculate)