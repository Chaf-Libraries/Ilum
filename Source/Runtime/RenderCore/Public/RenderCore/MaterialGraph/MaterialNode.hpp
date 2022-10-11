#pragma once

#include <Core/Macro.hpp>

namespace Ilum
{
struct MaterialGraphDesc;

STRUCT(MaterialNodePin, Enable)
{
	ENUM(Attribute, Enable){
	    Input,
	    Output};

	ENUM(Type, Enable) :
	    uint64_t{
	        Float        = 1,
	        Float2       = 1 << 2,
	        Float3       = 1 << 3,
	        Float4       = 1 << 4,
	        Int          = 1 << 5,
	        Int2         = 1 << 6,
	        Int3         = 1 << 7,
	        Int4         = 1 << 8,
	        Uint         = 1 << 9,
	        Uint2        = 1 << 10,
	        Uint3        = 1 << 11,
	        Uint4        = 1 << 12,
	        Bool         = 1 << 13,
	        Bool2        = 1 << 14,
	        Bool3        = 1 << 15,
	        Bool4        = 1 << 16,
	        Texture2D    = 1 << 17,
	        Texture3D    = 1 << 18,
	        TextureCube  = 1 << 19,
	        SamplerState = 1 << 20,
	        BSDF         = 1 << 21,
	    };

	Type          type;
	Attribute     attribute;
	std::string   name;
	size_t        handle;
	rttr::variant data;
};

DEFINE_ENUMCLASS_OPERATION(MaterialNodePin::Type)

STRUCT(MaterialNodeDesc, Enable)
{
	std::string name;

	size_t handle;

	std::map<std::string, MaterialNodePin> pins;
	std::map<size_t, std::string>          pin_index;

	rttr::variant data;

	MaterialNodeDesc &AddPin(size_t & handle, const std::string &name, MaterialNodePin::Type type, MaterialNodePin::Attribute attribute, rttr::variant data = {})
	{
		MaterialNodePin pin{type, attribute, name, handle, data};
		pin_index.emplace(handle, name);
		pins.emplace(name, pin);
		handle++;
		return *this;
	}

	const MaterialNodePin &GetPin(const std::string &name) const
	{
		return pins.at(name);
	}

	const MaterialNodePin &GetPin(size_t handle) const
	{
		return pins.at(pin_index.at(handle));
	}

	template <typename T>
	MaterialNodeDesc &SetName()
	{
		name = rttr::type::get<T>().get_name().to_string();
		return *this;
	}

	MaterialNodeDesc &SetData(rttr::variant data)
	{
		this->data = data;
		return *this;
	}
};

struct MaterialEmitInfo
{
	std::vector<std::string> definitions;
	std::string type_name;
	std::string name;

	std::map<std::string, MaterialNodePin::Type> uniform_resource;
	std::unordered_set<std::string> includes;

	bool IsRecursion() const
	{
		return name.empty();
	}
};

STRUCT(MaterialNode, Enable)
{
	virtual MaterialNodeDesc Create(size_t & handle) = 0;

	virtual bool Validate(const MaterialNodeDesc &node, MaterialGraphDesc &graph)
	{
		return true;
	}

	virtual void EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo& info) = 0;

	RTTR_ENABLE();
};
}        // namespace Ilum