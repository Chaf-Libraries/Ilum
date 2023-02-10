#include "RenderPass.hpp"

namespace Ilum
{
RenderPassDesc &RenderPassDesc::SetName(const std::string &name)
{
	m_name = name;
	return *this;
}

RenderPassDesc &RenderPassDesc::SetCategory(const std::string &category)
{
	m_category = category;
	return *this;
}

RenderPassDesc &RenderPassDesc::SetHandle(size_t handle)
{
	m_handle = handle;
	return *this;
}

RenderPassDesc &RenderPassDesc::WriteTexture2D(size_t handle, const std::string &name, uint32_t width, uint32_t height, RHIFormat format, RHIResourceState resource_state)
{
	RenderPassPin pin;
	pin.type           = RenderPassPin::Type::Texture;
	pin.attribute      = RenderPassPin::Attribute::Output;
	pin.name           = name;
	pin.handle         = handle;
	pin.texture        = TextureDesc{name, width, height, 1, 1, 1, 1, format, RHITextureUsage::Undefined};
	pin.resource_state = resource_state;

	m_pins.emplace(handle, pin);
	m_pin_indices.emplace(name, handle);
	return *this;
}

RenderPassDesc &RenderPassDesc::ReadTexture2D(size_t handle, const std::string &name, RHIResourceState resource_state)
{
	RenderPassPin pin;
	pin.type           = RenderPassPin::Type::Texture;
	pin.attribute      = RenderPassPin::Attribute::Input;
	pin.name           = name;
	pin.handle         = handle;
	pin.texture        = TextureDesc{name, 1, 1, 1, 1, 1, 1, RHIFormat::Undefined, RHITextureUsage::Undefined};
	pin.resource_state = resource_state;

	m_pins.emplace(handle, pin);
	m_pin_indices.emplace(name, handle);
	return *this;
}

RenderPassDesc &RenderPassDesc::WriteBuffer(size_t handle, const std::string &name, size_t size, RHIResourceState resource_state)
{
	RenderPassPin pin;
	pin.type           = RenderPassPin::Type::Buffer;
	pin.attribute      = RenderPassPin::Attribute::Output;
	pin.name           = name;
	pin.handle         = handle;
	pin.buffer         = BufferDesc{name, RHIBufferUsage::UnorderedAccess | RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::GPU_Only, size, 0, 0};
	pin.resource_state = resource_state;

	m_pins.emplace(handle, pin);
	m_pin_indices.emplace(name, handle);
	return *this;
}

RenderPassDesc &RenderPassDesc::ReadBuffer(size_t handle, const std::string &name, RHIResourceState resource_state)
{
	RenderPassPin pin;
	pin.type           = RenderPassPin::Type::Buffer;
	pin.attribute      = RenderPassPin::Attribute::Input;
	pin.name           = name;
	pin.handle         = handle;
	pin.buffer         = BufferDesc{name, RHIBufferUsage::UnorderedAccess | RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::GPU_Only, 0, 0, 0};
	pin.resource_state = resource_state;

	m_pins.emplace(handle, pin);
	m_pin_indices.emplace(name, handle);
	return *this;
}

const RenderPassPin &RenderPassDesc::GetPin(size_t handle) const
{
	return m_pins.at(handle);
}

RenderPassPin &RenderPassDesc::GetPin(size_t handle)
{
	return m_pins.at(handle);
}

RenderPassPin &RenderPassDesc::GetPin(const std::string &name)
{
	return m_pins.at(m_pin_indices.at(name));
}

const RenderPassPin &RenderPassDesc::GetPin(const std::string &name) const
{
	return m_pins.at(m_pin_indices.at(name));
}

RenderPassDesc &RenderPassDesc::SetConfig(Variant config)
{
	m_config = config;
	return *this;
}

const Variant &RenderPassDesc::GetConfig() const
{
	return m_config;
}

const std::string &RenderPassDesc::GetName() const
{
	return m_name;
}

const std::string &RenderPassDesc::GetCategory() const
{
	return m_category;
}

std::map<size_t, RenderPassPin> &RenderPassDesc::GetPins()
{
	return m_pins;
}

size_t RenderPassDesc::GetHandle() const
{
	return m_handle;
}

BindPoint RenderPassDesc::GetBindPoint() const
{
	return m_bind_point;
}

RenderPassDesc &RenderPassDesc::SetBindPoint(BindPoint bind_point)
{
	m_bind_point = bind_point;
	return *this;
}
}        // namespace Ilum