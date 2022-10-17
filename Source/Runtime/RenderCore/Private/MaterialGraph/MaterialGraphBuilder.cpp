#include "MaterialGraph/MaterialGraphBuilder.hpp"
#include "MaterialGraph/MaterialGraph.hpp"
#include "MaterialGraph/MaterialNode.hpp"
#include "MaterialGraph/MaterialNode/OutputNode.hpp"

#include <Core/Path.hpp>
#include <RHI/RHIContext.hpp>

#include <mustache.hpp>

namespace Ilum
{
MaterialGraphBuilder::MaterialGraphBuilder(RHIContext *rhi_context) :
    p_rhi_context(rhi_context)
{
}

void MaterialGraphBuilder::Compile(MaterialGraph *graph)
{
	// Check Output Node
	size_t output_handle = 0;
	{
		uint32_t output_count = 0;
		for (auto &[handle, node] : graph->GetDesc().nodes)
		{
			if (node.name == rttr::type::get<MGNode::MaterialOutput>().get_name().to_string())
			{
				output_count++;
				output_handle = handle;
			}
		}
		if (output_count == 0)
		{
			LOG_ERROR("Material Graph must have one output node!");
			return;
		}
		else if (output_count > 1)
		{
			LOG_ERROR("Material Graph must have only one output node!");
			return;
		}
	}

	// Validate
	{
		ShaderValidateContext validate_context;

		const auto &output_node_desc = graph->GetDesc().nodes.at(output_handle);
		auto        output_node      = rttr::type::get_by_name(output_node_desc.name).create();

		output_node.get_type().get_method("Validate").invoke(output_node, output_node_desc, graph, validate_context);

		if (!validate_context.error_infos.empty())
		{
			for (auto &info : validate_context.error_infos)
			{
				LOG_ERROR(info);
			}
			return;
		}

		if (validate_context.valid_nodes.size() != graph->GetDesc().nodes.size())
		{
			std::vector<size_t> node_removal;
			for (auto &[handle, node] : graph->GetDesc().nodes)
			{
				if (validate_context.valid_nodes.find(handle) == validate_context.valid_nodes.end())
				{
					node_removal.emplace_back(handle);
					LOG_INFO("Node {} is never used, culling it", node.name);
				}
			}
			for (auto &handle : node_removal)
			{
				graph->GetDesc().EraseNode(handle);
			}
		}
	}

	// Emit Shader
	{
		ShaderEmitContext emit_context;

		const auto &output_node_desc = graph->GetDesc().nodes.at(output_handle);
		auto        output_node      = rttr::type::get_by_name(output_node_desc.name).create();

		output_node.get_type().get_method("EmitShader").invoke(output_node, output_node_desc, graph, emit_context);
	}
}
}        // namespace Ilum