#include "ConvertNode.hpp"
#include "MaterialGraph/MaterialGraph.hpp"

namespace Ilum::MGNode
{
MaterialNodeDesc ScalarCalculation::Create(size_t &handle)
{
	MaterialNodeDesc desc;
	return desc.SetName<ScalarCalculation>()
	    .AddPin(handle, "X", MaterialNodePin::Type::Float, MaterialNodePin::Attribute::Input, float(0))
	    .AddPin(handle, "Y", MaterialNodePin::Type::Float, MaterialNodePin::Attribute::Input, float(0))
	    .AddPin(handle, "Out", MaterialNodePin::Type::Float, MaterialNodePin::Attribute::Output)
	    .SetData(ScalarCalculationType());
}

void ScalarCalculation::Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context)
{
	if (context.finish.find(node.handle) != context.finish.end())
	{
		return;
	}

	if (graph->GetDesc().HasLink(node.GetPin("X").handle))
	{
		graph->Validate(graph->GetDesc().LinkFrom(node.GetPin("X").handle), context);
	}

	if (graph->GetDesc().HasLink(node.GetPin("Y").handle))
	{
		graph->Validate(graph->GetDesc().LinkFrom(node.GetPin("Y").handle), context);
	}

	context.valid_nodes.insert(node.handle);

	context.finish.insert(node.handle);
}

void ScalarCalculation::Update(MaterialNodeDesc &node)
{
	Type type = node.data.convert<Type>();
	switch (type)
	{
		case Type::Addition:
		case Type::Substrate:
		case Type::Multiplication:
		case Type::Division:
		case Type::Maximum:
		case Type::Minimum:
		case Type::Greater:
		case Type::Less:
		case Type::Square:
		case Type::Atan2:
			node.GetPin("Y").enable = true;
			break;
		case Type::Log:
		case Type::Exp:
		case Type::Sqrt:
		case Type::Rcp:
		case Type::Abs:
		case Type::Sign:
		case Type::Sin:
		case Type::Cos:
		case Type::Tan:
		case Type::Asin:
		case Type::Acos:
		case Type::Atan:
		case Type::Sinh:
		case Type::Cosh:
		case Type::Tanh:
			node.GetPin("Y").enable = false;
			break;
		default:
			break;
	}
}

void ScalarCalculation::EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context)
{
	if (context.finish.find(desc.handle) != context.finish.end())
	{
		return;
	}

	std::string lhs, rhs;

	if (graph->GetDesc().HasLink(desc.GetPin("X").handle))
	{
		lhs = fmt::format("_S{}", graph->GetDesc().LinkFrom(desc.GetPin("X").handle));
		graph->EmitShader(graph->GetDesc().LinkFrom(desc.GetPin("X").handle), context);
	}
	else
	{
		auto pin = desc.GetPin("X").data.convert<float>();
		lhs      = fmt::format("{}", pin);
	}

	if (graph->GetDesc().HasLink(desc.GetPin("Y").handle))
	{
		rhs = fmt::format("_S{}", graph->GetDesc().LinkFrom(desc.GetPin("Y").handle));
		graph->EmitShader(graph->GetDesc().LinkFrom(desc.GetPin("Y").handle), context);
	}
	else
	{
		auto pin = desc.GetPin("Y").data.convert<float>();
		rhs      = fmt::format("{}", pin);
	}

	context.declarations.emplace_back(fmt::format("float _S{};", desc.GetPin("Out").handle));

	Type type = desc.data.convert<Type>();
	switch (type)
	{
		case Type::Addition:
			context.definitions.emplace_back(fmt::format("_S{} = {} + {};", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Substrate:
			context.definitions.emplace_back(fmt::format("_S{} = {} - {};", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Multiplication:
			context.definitions.emplace_back(fmt::format("_S{} = {} * {};", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Division:
			context.definitions.emplace_back(fmt::format("_S{} = {} / {};", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Maximum:
			context.definitions.emplace_back(fmt::format("_S{} = max({}, {});", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Minimum:
			context.definitions.emplace_back(fmt::format("_S{} = min({}, {});", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Greater:
			context.definitions.emplace_back(fmt::format("_S{} = float({} > {});", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Less:
			context.definitions.emplace_back(fmt::format("_S{} = float({} < {});", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Square:
			context.definitions.emplace_back(fmt::format("_S{} = {} * {};", desc.GetPin("Out").handle, lhs, lhs));
			break;
		case Type::Log:
			context.definitions.emplace_back(fmt::format("_S{} = log({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Exp:
			context.definitions.emplace_back(fmt::format("_S{} = exp({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Sqrt:
			context.definitions.emplace_back(fmt::format("_S{} = sqrt({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Rcp:
			context.definitions.emplace_back(fmt::format("_S{} = rcp({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Abs:
			context.definitions.emplace_back(fmt::format("_S{} = abs({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Sign:
			context.definitions.emplace_back(fmt::format("_S{} = sgn({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Sin:
			context.definitions.emplace_back(fmt::format("_S{} = sin({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Cos:
			context.definitions.emplace_back(fmt::format("_S{} = cos({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Tan:
			context.definitions.emplace_back(fmt::format("_S{} = tan({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Asin:
			context.definitions.emplace_back(fmt::format("_S{} = asin({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Acos:
			context.definitions.emplace_back(fmt::format("_S{} = acos({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Atan:
			context.definitions.emplace_back(fmt::format("_S{} = atan({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Atan2:
			context.definitions.emplace_back(fmt::format("_S{} = atan2({}, {});", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Sinh:
			context.definitions.emplace_back(fmt::format("_S{} = sinh({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Cosh:
			context.definitions.emplace_back(fmt::format("_S{} = cosh({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Tanh:
			context.definitions.emplace_back(fmt::format("_S{} = tanh({});", desc.GetPin("Out").handle, lhs));
			break;
		default:
			break;
	}

	context.finish.emplace(desc.handle);
}

MaterialNodeDesc VectorCalculation::Create(size_t &handle)
{
	MaterialNodeDesc desc;
	return desc.SetName<VectorCalculation>()
	    .AddPin(handle, "X", MaterialNodePin::Type::Float3, MaterialNodePin::Attribute::Input, glm::vec3(0))
	    .AddPin(handle, "Y", MaterialNodePin::Type::Float3, MaterialNodePin::Attribute::Input, glm::vec3(0))
	    .AddPin(handle, "Out", MaterialNodePin::Type::Float3, MaterialNodePin::Attribute::Output)
	    .SetData(VectorCalculationType());
}

void VectorCalculation::Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context)
{
	if (context.finish.find(node.handle) != context.finish.end())
	{
		return;
	}

	if (graph->GetDesc().HasLink(node.GetPin("X").handle))
	{
		graph->Validate(graph->GetDesc().LinkFrom(node.GetPin("X").handle), context);
	}

	if (graph->GetDesc().HasLink(node.GetPin("Y").handle))
	{
		graph->Validate(graph->GetDesc().LinkFrom(node.GetPin("Y").handle), context);
	}

	context.valid_nodes.insert(node.handle);

	context.finish.insert(node.handle);
}

void VectorCalculation::Update(MaterialNodeDesc &node)
{
	Type type = node.data.convert<Type>();
	switch (type)
	{
		case Type::Scale:
			node.GetPin("Y").enable = true;
			if (node.GetPin("Y").type != MaterialNodePin::Type::Float)
			{
				node.GetPin("Y").type = MaterialNodePin::Type::Float;
				node.GetPin("Y").data = float(0);
			}
			break;
		case Type::Length:
			node.GetPin("Y").enable = false;
			node.GetPin("Out").type = MaterialNodePin::Type::Float;
			break;
		case Type::Distance:
		case Type::Dot:
			node.GetPin("Y").enable = true;
			if (node.GetPin("Y").type != MaterialNodePin::Type::Float3)
			{
				node.GetPin("Y").type = MaterialNodePin::Type::Float3;
				node.GetPin("Y").data = glm::vec3(0);
			}
			node.GetPin("Out").type = MaterialNodePin::Type::Float;
			break;
		case Type::Cross:
		case Type::Addition:
		case Type::Substrate:
		case Type::Multiplication:
		case Type::Division:
		case Type::Maximum:
		case Type::Minimum:
			node.GetPin("Y").enable = true;
			if (node.GetPin("Y").type != MaterialNodePin::Type::Float3)
			{
				node.GetPin("Y").type = MaterialNodePin::Type::Float3;
				node.GetPin("Y").data = glm::vec3(0);
			}
			node.GetPin("Out").type = MaterialNodePin::Type::Float3;
			break;
		case Type::Sin:
		case Type::Cos:
		case Type::Tan:
		case Type::Abs:
			node.GetPin("Y").enable = false;
			node.GetPin("Out").type = MaterialNodePin::Type::Float3;
			break;
		case Type::Normalize:
			node.GetPin("Y").enable = false;
			node.GetPin("Out").type = MaterialNodePin::Type::Float;
			break;
		default:
			break;
	}
}

void VectorCalculation::EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context)
{
	if (context.finish.find(desc.handle) != context.finish.end())
	{
		return;
	}

	std::string lhs, rhs;

	if (graph->GetDesc().HasLink(desc.GetPin("X").handle))
	{
		lhs = fmt::format("_S{}", graph->GetDesc().LinkFrom(desc.GetPin("X").handle));
		graph->EmitShader(graph->GetDesc().LinkFrom(desc.GetPin("X").handle), context);
	}
	else
	{
		auto pin = desc.GetPin("X").data.convert<glm::vec3>();
		lhs      = fmt::format("float3({}, {}, {})", pin.x, pin.y, pin.z);
	}

	if (graph->GetDesc().HasLink(desc.GetPin("Y").handle))
	{
		rhs = fmt::format("_S{}", graph->GetDesc().LinkFrom(desc.GetPin("Y").handle));
		graph->EmitShader(graph->GetDesc().LinkFrom(desc.GetPin("Y").handle), context);
	}
	else
	{
		if (desc.GetPin("Y").type & MaterialNodePin::Type::Float3)
		{
			auto pin = desc.GetPin("Y").data.convert<glm::vec3>();
			rhs      = fmt::format("float3({}, {}, {})", pin.x, pin.y, pin.z);
		}
		else if (desc.GetPin("Y").type & MaterialNodePin::Type::Float)
		{
			auto pin = desc.GetPin("Y").data.convert<float>();
			rhs      = fmt::format("{}", pin);
		}
	}

	if (desc.GetPin("Out").type & MaterialNodePin::Type::Float)
	{
		context.declarations.emplace_back(fmt::format("float _S{};", desc.GetPin("Out").handle));
	}
	else if (desc.GetPin("Out").type & MaterialNodePin::Type::Float3)
	{
		context.declarations.emplace_back(fmt::format("float3 _S{};", desc.GetPin("Out").handle));
	}

	Type type = desc.data.convert<Type>();
	switch (type)
	{
		case Type::Scale:
			context.definitions.emplace_back(fmt::format("_S{} = {} * {};", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Length:
			context.definitions.emplace_back(fmt::format("_S{} = length({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Distance:
			context.definitions.emplace_back(fmt::format("_S{} = distance({}, {});", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Dot:
			context.definitions.emplace_back(fmt::format("_S{} = dot({}, {});", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Cross:
			context.definitions.emplace_back(fmt::format("_S{} = cross({}, {});", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Addition:
			context.definitions.emplace_back(fmt::format("_S{} = {} + {};", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Substrate:
			context.definitions.emplace_back(fmt::format("_S{} = {} - {};", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Multiplication:
			context.definitions.emplace_back(fmt::format("_S{} = {} * {};", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Division:
			context.definitions.emplace_back(fmt::format("_S{} = {} / {};", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Sin:
			context.definitions.emplace_back(fmt::format("_S{} = sin({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Cos:
			context.definitions.emplace_back(fmt::format("_S{} = cos({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Tan:
			context.definitions.emplace_back(fmt::format("_S{} = tan({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Maximum:
			context.definitions.emplace_back(fmt::format("_S{} = max({}, {});", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Minimum:
			context.definitions.emplace_back(fmt::format("_S{} = min({}, {});", desc.GetPin("Out").handle, lhs, rhs));
			break;
		case Type::Abs:
			context.definitions.emplace_back(fmt::format("_S{} = abs({});", desc.GetPin("Out").handle, lhs));
			break;
		case Type::Normalize:
			context.definitions.emplace_back(fmt::format("_S{} = normalize({});", desc.GetPin("Out").handle, lhs));
			break;
		default:
			break;
	}

	context.finish.emplace(desc.handle);
}

}        // namespace Ilum::MGNode