#include "ShaderNode.hpp"
#include "RenderCore/MaterialGraph/MaterialGraph.hpp"

#include <Core/Path.hpp>

#include "mustache.hpp"

namespace Ilum::MGNode
{
MaterialNodeDesc AddShader::Create(size_t &handle)
{
	MaterialNodeDesc desc;
	return desc.SetName<AddShader>()
	    .AddPin(handle, "BSDF A", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Input)
	    .AddPin(handle, "BSDF B", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Input)
	    .AddPin(handle, "BSDF", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Output);
}

void AddShader::Update(MaterialNodeDesc &node)
{
}

void AddShader::Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context)
{
	if (context.finish.find(node.handle) != context.finish.end())
	{
		return;
	}

	if (graph->GetDesc().HasLink(node.GetPin("BSDF A").handle))
	{
		graph->Validate(graph->GetDesc().LinkFrom(node.GetPin("BSDF A").handle), context);
	}

	if (graph->GetDesc().HasLink(node.GetPin("BSDF B").handle))
	{
		graph->Validate(graph->GetDesc().LinkFrom(node.GetPin("BSDF B").handle), context);
	}

	context.valid_nodes.insert(node.handle);

	context.finish.insert(node.handle);
}

void AddShader::EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context)
{
}

MaterialNodeDesc MixShader::Create(size_t &handle)
{
	MaterialNodeDesc desc;
	return desc.SetName<AddShader>()
	    .AddPin(handle, "BSDF A", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Input)
	    .AddPin(handle, "BSDF B", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Input)
	    .AddPin(handle, "Frac", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Input, Data{})
	    .AddPin(handle, "BSDF", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Output);
}

void MixShader::Update(MaterialNodeDesc &node)
{
}

void MixShader::Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context)
{
	if (context.finish.find(node.handle) != context.finish.end())
	{
		return;
	}

	if (graph->GetDesc().HasLink(node.GetPin("BSDF A").handle))
	{
		graph->Validate(graph->GetDesc().LinkFrom(node.GetPin("BSDF A").handle), context);
	}

	if (graph->GetDesc().HasLink(node.GetPin("BSDF B").handle))
	{
		graph->Validate(graph->GetDesc().LinkFrom(node.GetPin("BSDF B").handle), context);
	}

	if (graph->GetDesc().HasLink(node.GetPin("Frac").handle))
	{
		graph->Validate(graph->GetDesc().LinkFrom(node.GetPin("Frac").handle), context);
	}

	context.valid_nodes.insert(node.handle);

	context.finish.insert(node.handle);
}

void MixShader::EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context)
{
}

MaterialNodeDesc DiffuseBSDF::Create(size_t &handle)
{
	MaterialNodeDesc desc;
	return desc.SetName<DiffuseBSDF>()
	    .AddPin(handle, "Color", MaterialNodePin::Type::Float, MaterialNodePin::Attribute::Input, Color{})
	    .AddPin(handle, "Roughness", MaterialNodePin::Type::Float, MaterialNodePin::Attribute::Input, Roughness{})
	    .AddPin(handle, "BSDF", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Output);
}

void DiffuseBSDF::Update(MaterialNodeDesc &node)
{
}

void DiffuseBSDF::Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context)
{
	if (context.finish.find(node.handle) != context.finish.end())
	{
		return;
	}

	if (graph->GetDesc().HasLink(node.GetPin("Color").handle))
	{
		graph->Validate(graph->GetDesc().LinkFrom(node.GetPin("Color").handle), context);
	}

	if (graph->GetDesc().HasLink(node.GetPin("Roughness").handle))
	{
		graph->Validate(graph->GetDesc().LinkFrom(node.GetPin("Roughness").handle), context);
	}

	context.valid_nodes.insert(node.handle);

	context.finish.insert(node.handle);
}

void DiffuseBSDF::EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context)
{
	if (context.finish.find(desc.handle) != context.finish.end())
	{
		return;
	}

	std::string color, roughness;

	if (graph->GetDesc().HasLink(desc.GetPin("Color").handle))
	{
		color = fmt::format("_S{}", graph->GetDesc().LinkFrom(desc.GetPin("Color").handle));
		graph->EmitShader(graph->GetDesc().LinkFrom(desc.GetPin("Color").handle), context);
	}
	else
	{
		auto pin = desc.GetPin("Color").data.convert<Color>();
		color    = fmt::format("float3({}, {}, {})", pin.color.x, pin.color.y, pin.color.z);
	}

	if (graph->GetDesc().HasLink(desc.GetPin("Roughness").handle))
	{
		roughness = fmt::format("_S{}", graph->GetDesc().LinkFrom(desc.GetPin("Roughness").handle));
		graph->EmitShader(graph->GetDesc().LinkFrom(desc.GetPin("Roughness").handle), context);
	}
	else
	{
		auto pin  = desc.GetPin("Roughness").data.convert<Roughness>();
		roughness = fmt::format("{}", pin.roughness);
	}

	std::string bxdf_name = fmt::format("_S{}", desc.GetPin("BSDF").handle);

	context.headers.insert("Material/BxDF/DiffuseBSDF.hlsli");

	std::vector<uint8_t> shader_data;
	Path::GetInstance().Read("Source/Shaders/Material/BxDF.hlsli", shader_data);
	std::string shader(shader_data.begin(), shader_data.end());

	kainjow::mustache::mustache mustache = {shader};
	kainjow::mustache::data     mustache_data{kainjow::mustache::data::type::object};

	mustache_data["BxDFName"]    = bxdf_name;
	mustache_data["EvalFunc"]    = fmt::format("DiffuseBSDF::Eval({}, {}, wo, wi)", color, roughness);
	mustache_data["PdfFunc"]     = fmt::format("DiffuseBSDF::Pdf({}, {}, wo, wi)", color, roughness);
	mustache_data["SamplefFunc"] = fmt::format("DiffuseBSDF::Samplef({}, {}, wo, u, wi, pdf)", color, roughness);

	context.functions.emplace_back(std::string(mustache.render(mustache_data).c_str()));
	context.finish.emplace(desc.handle);
}
}        // namespace Ilum::MGNode