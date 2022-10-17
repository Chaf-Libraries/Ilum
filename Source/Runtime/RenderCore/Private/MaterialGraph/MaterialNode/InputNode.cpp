#include "InputNode.hpp"

namespace Ilum::MGNode
{
MaterialNodeDesc RGB::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	return desc
	    .SetName<RGB>()
	    .AddPin(handle, "Color", MaterialNodePin::Type::Float, MaterialNodePin::Attribute::Output)
	    .SetData(Data());
}

void RGB::Update(MaterialNodeDesc &node)
{
}

void RGB::Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context)
{
	if (context.finish.find(node.handle) != context.finish.end())
	{
		return;
	}
	context.valid_nodes.insert(node.handle);
	context.finish.emplace(node.handle);
}

void RGB::EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context)
{
	if (context.finish.find(desc.handle) != context.finish.end())
	{
		return;
	}

	Data color = desc.data.convert<Data>();
	context.declarations.emplace_back(fmt::format("float3 _S{};", desc.GetPin("Color").handle));
	context.definitions.emplace_back(fmt::format("_S{} = float3({}, {}, {});", desc.GetPin("Color").handle, color.color.x, color.color.y, color.color.z));

	context.finish.emplace(desc.handle);
}

MaterialNodeDesc ObjectInfo::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	return desc
	    .SetName<ObjectInfo>()
	    .AddPin(handle, "Position", MaterialNodePin::Type::Float, MaterialNodePin::Attribute::Output)
	    .AddPin(handle, "Instance ID", MaterialNodePin::Type::Float, MaterialNodePin::Attribute::Output)
	    .AddPin(handle, "Material ID", MaterialNodePin::Type::Float, MaterialNodePin::Attribute::Output)
	    .AddPin(handle, "Meshlet ID", MaterialNodePin::Type::Float, MaterialNodePin::Attribute::Output);
}

void ObjectInfo::Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context)
{
}

void ObjectInfo::EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context)
{
	if (context.finish.find(desc.handle) != context.finish.end())
	{
		return;
	}

	context.finish.emplace(desc.handle);
}

void ObjectInfo::Update(MaterialNodeDesc &node)
{
}
}        // namespace Ilum::MGNode