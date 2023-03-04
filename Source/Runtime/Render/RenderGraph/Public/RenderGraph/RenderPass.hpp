#pragma once

#include "Precompile.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class Editor;

enum class BindPoint
{
	None,
	Rasterization,
	Compute,
	RayTracing,
	CUDA
};

struct  RenderPassPin
{
	enum class Type
	{
		Unknown,
		Texture,
		Buffer
	};

	enum class Attribute
	{
		Input,
		Output
	};

	// Common
	Type      type;
	Attribute attribute;

	std::string name;
	size_t      handle;

	TextureDesc texture;
	BufferDesc  buffer;

	RHIResourceState resource_state;

	template <typename Archive>
	void serialize(Archive &archive)
	{
		archive(type, attribute, name, handle, texture, buffer, resource_state);
	}
};

class  RenderPassDesc
{
  public:
	RenderPassDesc() = default;

	~RenderPassDesc() = default;

	RenderPassDesc &SetName(const std::string &name);

	RenderPassDesc &SetCategory(const std::string &category);

	RenderPassDesc &SetHandle(size_t handle);

	RenderPassDesc &WriteTexture2D(size_t handle, const std::string &name, uint32_t width, uint32_t height, RHIFormat format, RHIResourceState resource_state);

	RenderPassDesc &WriteTexture2D(size_t handle, const std::string &name, uint32_t width, uint32_t height, uint32_t layer, RHIFormat format, RHIResourceState resource_state);

	RenderPassDesc &ReadTexture2D(size_t handle, const std::string &name, RHIResourceState resource_state);

	RenderPassDesc &WriteBuffer(size_t handle, const std::string &name, size_t size, RHIResourceState resource_state);

	RenderPassDesc &ReadBuffer(size_t handle, const std::string &name, RHIResourceState resource_state);

	const RenderPassPin &GetPin(size_t handle) const;

	RenderPassPin &GetPin(size_t handle);

	RenderPassPin &GetPin(const std::string &name);

	const RenderPassPin &GetPin(const std::string &name) const;

	RenderPassDesc &SetConfig(Variant config);

	const Variant &GetConfig() const;

	const std::string &GetName() const;

	const std::string &GetCategory() const;

	std::map<size_t, RenderPassPin> &GetPins();

	size_t GetHandle() const;

	BindPoint GetBindPoint() const;

	RenderPassDesc &SetBindPoint(BindPoint bind_point);

	template <typename Archive>
	void serialize(Archive &archive)
	{
		archive(m_name, m_category, m_bind_point, m_handle, m_pins, m_pin_indices, m_config);
	}

  private:
	std::string m_name;
	std::string m_category;

	BindPoint m_bind_point = BindPoint::None;

	size_t m_handle = ~0ull;

	std::map<size_t, RenderPassPin> m_pins;
	std::map<std::string, size_t>   m_pin_indices;

	Variant m_config;
};
}        // namespace Ilum