#include "BxDFNode.hpp"
#include "MaterialGraph/MaterialGraph.hpp"

#include <Core/Path.hpp>

#include <mustache.hpp>

namespace Ilum::MGNode
{
MaterialNodeDesc BlendBxDFNode::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	desc.SetName<BlendBxDFNode>()
	    .AddPin(handle, "LHS", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Input)
	    .AddPin(handle, "Out", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Output)
	    .AddPin(handle, "RHS", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Input)
	    .AddPin(handle, "Weight", MaterialNodePin::Type::Float, MaterialNodePin::Attribute::Input, BlendWeight{});

	return desc;
}

void BlendBxDFNode::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info)
{
	const auto &lhs_desc = graph.GetNode(graph.LinkFrom(desc.GetPin("LHS").handle));
	auto        lhs_node = rttr::type::get_by_name(lhs_desc.name).create();
	lhs_node.get_type().get_method("EmitHLSL").invoke(lhs_node, lhs_desc, graph, info);

	std::string lhs_type = info.type_name;
	std::string lhs_name = info.name;

	const auto &rhs_desc = graph.GetNode(graph.LinkFrom(desc.GetPin("RHS").handle));
	auto        rhs_node = rttr::type::get_by_name(rhs_desc.name).create();
	rhs_node.get_type().get_method("EmitHLSL").invoke(rhs_node, rhs_desc, graph, info);

	std::string rhs_type = info.type_name;
	std::string rhs_name = info.name;

	info.type_name = "BlendBxDF" + std::to_string(desc.handle);

	{
		std::vector<uint8_t> material_data;
		Path::GetInstance().Read("Source/Shaders/Material/BlendBxDF.hlsli", material_data);
		std::string material_shader(material_data.begin(), material_data.end());

		kainjow::mustache::mustache mustache = {material_shader};

		kainjow::mustache::data mustache_data{kainjow::mustache::data::type::object};
		mustache_data["BxDFName"]  = info.type_name;
		mustache_data["BxDFTypeA"] = lhs_type;
		mustache_data["BxDFTypeB"] = rhs_type;
		if (graph.HasLink(desc.GetPin("Weight").handle))
		{
			MaterialEmitInfo weight_info;
			const auto      &weight_desc = graph.GetNode(graph.LinkFrom(desc.GetPin("Weight").handle));
			auto             weight_node = rttr::type::get_by_name(weight_desc.name).create();
			weight_node.get_type().get_method("EmitHLSL").invoke(weight_node, weight_desc, graph, weight_info);

			kainjow::mustache::data definitions{kainjow::mustache::data::type::list};
			for (auto &weight_defintion : weight_info.definitions)
			{
				kainjow::mustache::data definition{kainjow::mustache::data::type::object};
				definition["Definition"] = weight_defintion;
				definitions << definition;
			}
			mustache_data.set("Definitions", definitions);
			if (weight_info.IsExpression(graph.LinkFrom(desc.GetPin("Weight").handle)))
			{
				mustache_data["Weight"] = fmt::format("CastFloat({})", weight_info.expression.at(graph.LinkFrom(desc.GetPin("Weight").handle)));
			}
			else
			{
				mustache_data["Weight"] = "S" + std::to_string(graph.LinkFrom(desc.GetPin("Weight").handle));
			}
		}
		else
		{
			mustache_data["Weight"] = std::to_string(desc.GetPin("Weight").data.convert<BlendWeight>().Weight);
		}

		info.definitions.push_back(std::string(mustache.render(mustache_data).c_str()));
	}
}
}        // namespace Ilum::MGNode