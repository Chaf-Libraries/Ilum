#pragma once

#include "LambertianReflection.hpp"
#include "MaterialGraph/MaterialGraph.hpp"

#include <Core/Path.hpp>

#include <mustache.hpp>

namespace Ilum::MGNode
{
MaterialNodeDesc LambertianReflection::Create(size_t &handle)
{
	return BxDFNode::Create(handle)
	    .AddPin(handle, "BaseColor", MaterialNodePin::Type::Float3, MaterialNodePin::Attribute::Input, BaseColor{})
	    .AddPin(handle, "Out", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Output);
}

void LambertianReflection::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info)
{
	info.includes.insert("Material/BxDF/LambertianReflection.hlsli");
	info.type_name = "LambertianReflection" + std::to_string(desc.handle);
	

	{
		std::vector<uint8_t> material_data;
		Path::GetInstance().Read("Source/Shaders/Material/BxDF.hlsli", material_data);
		std::string material_shader(material_data.begin(), material_data.end());

		kainjow::mustache::mustache mustache = {material_shader};

		kainjow::mustache::data mustache_data{kainjow::mustache::data::type::object};
		mustache_data["BxDFName"] = info.type_name;
		mustache_data["BxDFType"] = "LambertianReflection";
		mustache_data["Parameter"] = "float3(1.0, 1.0, 1.0)";

		info.definitions.push_back(std::string(mustache.render(mustache_data).c_str()));
	}


}
}        // namespace Ilum::MGNode