#include "MaterialNode.hpp"
#include "MaterialGraph.hpp"

namespace Ilum
{
MaterialNodeDesc &MaterialNodeDesc::SetName(const std::string &name)
{
	m_name = name;
	return *this;
}

MaterialNodeDesc &MaterialNodeDesc::SetCategory(const std::string &category)
{
	m_category = category;
	return *this;
}

MaterialNodeDesc &MaterialNodeDesc::SetHandle(size_t handle)
{
	m_handle = handle;
	return *this;
}

MaterialNodeDesc &MaterialNodeDesc::Input(size_t handle, const std::string &name, MaterialNodePin::Type type, MaterialNodePin::Type accept, Variant &&variant)
{
	MaterialNodePin pin{type, accept == MaterialNodePin::Type::Unknown ? type : accept, MaterialNodePin::Attribute::Input, name, handle, std::move(variant)};
	m_pins.emplace(handle, pin);
	m_pin_indices.emplace(name, handle);
	return *this;
}

MaterialNodeDesc &MaterialNodeDesc::Output(size_t handle, const std::string &name, MaterialNodePin::Type type, Variant &&variant)
{
	MaterialNodePin pin{type, MaterialNodePin::Type::Unknown, MaterialNodePin::Attribute::Output, name, handle, std::move(variant)};
	m_pins.emplace(handle, pin);
	m_pin_indices.emplace(name, handle);
	return *this;
}

MaterialNodePin &MaterialNodeDesc::GetPin(size_t handle)
{
	return m_pins.at(handle);
}

const MaterialNodePin &MaterialNodeDesc::GetPin(size_t handle) const
{
	return m_pins.at(handle);
}

MaterialNodePin &MaterialNodeDesc::GetPin(const std::string &name)
{
	return m_pins.at(m_pin_indices.at(name));
}

const MaterialNodePin &MaterialNodeDesc::GetPin(const std::string &name) const
{
	return m_pins.at(m_pin_indices.at(name));
}

MaterialNodeDesc &MaterialNodeDesc::SetVariant(Variant variant)
{
	m_variant = variant;
	return *this;
}

const Variant &MaterialNodeDesc::GetVariant() const
{
	return m_variant;
}

const std::string &MaterialNodeDesc::GetName() const
{
	return m_name;
}

const std::string &MaterialNodeDesc::GetCategory() const
{
	return m_category;
}

const std::map<size_t, MaterialNodePin> &MaterialNodeDesc::GetPins() const
{
	return m_pins;
}

size_t MaterialNodeDesc::GetHandle() const
{
	return m_handle;
}

void MaterialNodeDesc::EmitHLSL(const MaterialGraphDesc &graph_desc, Renderer *renderer, MaterialCompilationContext *context) const
{
	PluginManager::GetInstance().Call(fmt::format("shared/Material/Material.{}.{}.dll", m_category, m_name), "EmitHLSL", *this, graph_desc, renderer, context);
}
}        // namespace Ilum