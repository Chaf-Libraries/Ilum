#include "OutputNode.hpp"
#include "MaterialGraph/MaterialGraph.hpp"

#include <Core/Path.hpp>

#include <mustache.hpp>

namespace Ilum::MGNode
{
MaterialNodeDesc MaterialOutput::Create(size_t &handle)
{
	MaterialNodeDesc desc;
	return desc.SetName<MaterialOutput>()
	    .AddPin(handle, "Surface", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Input)
	    .AddPin(handle, "Volume", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Input);
}

void MaterialOutput::Update(MaterialNodeDesc &node)
{
}

void MaterialOutput::Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context)
{
	if (context.finish.find(node.handle) != context.finish.end())
	{
		return;
	}

	if (graph->GetDesc().HasLink(node.GetPin("Surface").handle))
	{
		graph->Validate(graph->GetDesc().LinkFrom(node.GetPin("Surface").handle), context);
	}

	if (graph->GetDesc().HasLink(node.GetPin("Volume").handle))
	{
		graph->Validate(graph->GetDesc().LinkFrom(node.GetPin("Volume").handle), context);
	}

	context.valid_nodes.insert(node.handle);

	context.finish.insert(node.handle);
}

void MaterialOutput::EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context)
{
	if (context.finish.find(desc.handle) != context.finish.end())
	{
		return;
	}

	std::string surface, volume;

	if (graph->GetDesc().HasLink(desc.GetPin("Surface").handle))
	{
		surface = fmt::format("_S{}", graph->GetDesc().LinkFrom(desc.GetPin("Surface").handle));
		graph->EmitShader(graph->GetDesc().LinkFrom(desc.GetPin("Surface").handle), context);
	}

	if (graph->GetDesc().HasLink(desc.GetPin("Volume").handle))
	{
		volume = fmt::format("_S{}", graph->GetDesc().LinkFrom(desc.GetPin("Volume").handle));
		graph->EmitShader(graph->GetDesc().LinkFrom(desc.GetPin("Volume").handle), context);
	}

	std::vector<uint8_t> material_data;
	Path::GetInstance().Read("Source/Shaders/Material.hlsli", material_data);
	std::string material(material_data.begin(), material_data.end());

	kainjow::mustache::mustache mustache = {material};
	kainjow::mustache::data     mustache_data{kainjow::mustache::data::type::object};

	mustache_data["SurfaceBSDF"] = surface;

	// Headers
	{
		kainjow::mustache::data headers{kainjow::mustache::data::type::list};
		for (auto &header_data : context.headers)
		{
			kainjow::mustache::data header{kainjow::mustache::data::type::object};
			header["Header"] = header_data;
			headers << header;
		}
		mustache_data.set("Headers", headers);
	}

	// Declarations
	{
		kainjow::mustache::data declarations{kainjow::mustache::data::type::list};
		for (auto &declaration_data : context.declarations)
		{
			kainjow::mustache::data declaration{kainjow::mustache::data::type::object};
			declaration["Declaration"] = declaration_data;
			declarations << declaration;
		}
		mustache_data.set("Declarations", declarations);
	}

	// Declarations
	{
		kainjow::mustache::data functions{kainjow::mustache::data::type::list};
		for (auto &function_data : context.functions)
		{
			kainjow::mustache::data function{kainjow::mustache::data::type::object};
			function["Function"] = function_data;
			functions << function;
		}
		mustache_data.set("Functions", functions);
	}

	// Definitions
	{
		kainjow::mustache::data definitions{kainjow::mustache::data::type::list};
		for (auto &definition_data : context.definitions)
		{
			kainjow::mustache::data definition{kainjow::mustache::data::type::object};
			definition["Definition"] = definition_data;
			definitions << definition;
		}
		mustache_data.set("Definitions", definitions);
	}

	context.result = std::string(mustache.render(mustache_data).c_str());
	context.finish.emplace(desc.handle);
}

}        // namespace Ilum::MGNode