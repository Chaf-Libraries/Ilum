#include "MaterialGraph/MaterialGraphBuilder.hpp"
#include "MaterialGraph/MaterialGraph.hpp"
#include "MaterialGraph/MaterialNode.hpp"
#include "MaterialGraph/MaterialNode/SemanticNode.hpp"

#include <Core/Path.hpp>
#include <RHI/RHIContext.hpp>

#include <mustache.hpp>

namespace Ilum
{
MaterialGraphBuilder::MaterialGraphBuilder(RHIContext *rhi_context) :
    p_rhi_context(rhi_context)
{
}

std::unique_ptr<MaterialGraph> MaterialGraphBuilder::Compile(MaterialGraphDesc &desc)
{
	// Check Output Node
	size_t output_handle = 0;
	{
		uint32_t output_count = 0;
		for (auto &[handle, node] : desc.nodes)
		{
			if (node.name == rttr::type::get<MGNode::OutputNode>().get_name().to_string())
			{
				output_count++;
				output_handle = handle;
			}
		}
		if (output_count == 0)
		{
			LOG_ERROR("Material Graph must have one output node!");
			return nullptr;
		}
		else if (output_count > 1)
		{
			LOG_ERROR("Material Graph must have only one output node!");
			return nullptr;
		}
	}

	const auto &output_node_desc = desc.nodes.at(output_handle);

	MaterialEmitInfo emit_info;

	auto output_node = rttr::type::get_by_name(output_node_desc.name).create();
	output_node.get_type().get_method("EmitHLSL").invoke(output_node, output_node_desc, desc, emit_info);

	std::vector<uint8_t> material_data;

	Path::GetInstance().Read("Source/Shaders/Material.hlsli", material_data);

	std::string material_shader(material_data.begin(), material_data.end());

	kainjow::mustache::mustache mustache = {material_shader};

	kainjow::mustache::data mustache_data{kainjow::mustache::data::type::object};
	mustache_data.set("BxDFType", emit_info.type_name);
	{
		kainjow::mustache::data includes{kainjow::mustache::data::type::list};
		for (auto &include_header : emit_info.includes)
		{
			kainjow::mustache::data include{kainjow::mustache::data::type::object};
			include["IncludePath"] = fmt::format("#include \"{}\"", include_header);
			includes << include;
		}
		mustache_data.set("Include", includes);
	}

	std::string shader = mustache.render(mustache_data);

	// surface_bsdf_node.get_type().get_method("EmitHLSL").invoke(surface_bsdf_node, surface_bsdf_info).to_string();

	return nullptr;
}
}        // namespace Ilum