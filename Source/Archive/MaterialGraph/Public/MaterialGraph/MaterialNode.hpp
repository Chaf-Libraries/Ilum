#pragma once

#include <Core/Macro.hpp>

namespace Ilum
{
class MaterialGraph;

STRUCT(MaterialNodePin, Enable)
{
	ENUM(Attribute, Enable){
	    Input,
	    Output};

	ENUM(Type, Enable) :
	    uint64_t{
	        Float  = 1,
	        Float3 = 1 << 1,
	        BSDF   = 1 << 2,
	    };

	Type type;

	Attribute attribute;

	std::string name;

	size_t handle;

	rttr::variant data;

	bool enable = true;
};

DEFINE_ENUMCLASS_OPERATION(MaterialNodePin::Type)

STRUCT(MaterialNodeDesc, Enable)
{
	std::string name;

	size_t handle;

	std::map<size_t, MaterialNodePin> pins;

	std::map<std::string, size_t> pin_index;

	rttr::variant data;

	MaterialNodeDesc &AddPin(size_t & handle, const std::string &name, MaterialNodePin::Type type, MaterialNodePin::Attribute attribute, rttr::variant data = {})
	{
		MaterialNodePin pin{type, attribute, name, handle, data};
		pin_index.emplace(name, handle);
		pins.emplace(handle, pin);
		handle++;
		return *this;
	}

	MaterialNodePin &GetPin(const std::string &name)
	{
		return pins.at(pin_index.at(name));
	}

	const MaterialNodePin &GetPin(const std::string &name) const
	{
		return pins.at(pin_index.at(name));
	}

	const MaterialNodePin &GetPin(size_t handle) const
	{
		return pins.at(handle);
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

struct ShaderEmitContext
{
	std::vector<std::string>        declarations;
	std::vector<std::string>        definitions;
	std::vector<std::string>        functions;
	std::unordered_set<std::string> headers;
	std::unordered_set<size_t>      finish;
	std::string                     result;
};

struct ShaderValidateContext
{
	std::unordered_set<size_t> valid_nodes;
	std::vector<std::string>   error_infos;
	std::unordered_set<size_t> finish;
};

STRUCT(MaterialNode, Enable)
{
	virtual MaterialNodeDesc Create(size_t & handle) = 0;

	virtual void Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context) = 0;

	virtual void Update(MaterialNodeDesc & node) = 0;

	virtual void EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context) = 0;

	RTTR_ENABLE();
};
}        // namespace Ilum