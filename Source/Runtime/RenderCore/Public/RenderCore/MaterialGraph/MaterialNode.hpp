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

	std::string          type;
	Attribute     attribute;
	std::string   name;
	size_t        handle;
	rttr::variant data;
};

STRUCT(MaterialNodeDesc, Enable)
{
	std::string name;

	std::map<std::string, MaterialNodePin> pins;
	std::map<size_t, std::string>          pin_index;

	rttr::variant data;

	MaterialNodeDesc &AddPin(size_t & handle, const std::string &name, std::string type, MaterialNodePin::Attribute attribute, rttr::variant data = {})
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
};

STRUCT(MaterialNode, Enable)
{
	virtual MaterialNodeDesc Create(size_t &handle) = 0;

	virtual std::string EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, std::string &source) = 0;

	RTTR_ENABLE();
};
}        // namespace Ilum