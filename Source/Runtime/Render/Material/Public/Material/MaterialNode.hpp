#pragma once

#include <Core/Core.hpp>

namespace Ilum
{
class MaterialGraphDesc;
class Editor;
class Renderer;
struct MaterialCompilationContext;

struct EXPORT_API MaterialNodePin
{
	enum class Type : uint64_t
	{
		Unknown = 0,
		Float   = 1,
		Float3  = 1 << 1,
		RGB     = 1 << 2,
		BSDF    = 1 << 4,
		Media    = 1 << 5,
	};

	enum class Attribute
	{
		Input,
		Output
	};

	Type        type;
	Type        accept;        //	for input pin
	Attribute   attribute;
	std::string name;
	size_t      handle;
	Variant     variant;

	bool enable = true;

	template <typename Archive>
	void serialize(Archive &archive)
	{
		archive(type, accept, attribute, name, handle, variant, enable);
	}
};

DEFINE_ENUMCLASS_OPERATION(MaterialNodePin::Type)

class EXPORT_API MaterialNodeDesc
{
  public:
	MaterialNodeDesc() = default;

	~MaterialNodeDesc() = default;

	MaterialNodeDesc &SetName(const std::string &name);

	MaterialNodeDesc &SetCategory(const std::string &category);

	MaterialNodeDesc &SetHandle(size_t handle);

	MaterialNodeDesc &Input(size_t handle, const std::string &name, MaterialNodePin::Type type, MaterialNodePin::Type accept = MaterialNodePin::Type::Unknown, Variant &&variant = {});

	MaterialNodeDesc &Output(size_t handle, const std::string &name, MaterialNodePin::Type type, Variant &&variant = {});

	const MaterialNodePin &GetPin(size_t handle) const;

	MaterialNodePin &GetPin(size_t handle);

	const MaterialNodePin &GetPin(const std::string &name) const;

	MaterialNodePin &GetPin(const std::string &name);

	MaterialNodeDesc &SetVariant(Variant variant);

	const Variant &GetVariant() const;

	const std::string &GetName() const;

	const std::string &GetCategory() const;

	const std::map<size_t, MaterialNodePin> &GetPins() const;

	size_t GetHandle() const;

	void EmitHLSL(const MaterialGraphDesc &graph_desc, Renderer *renderer, MaterialCompilationContext *context) const;

	template <typename Archive>
	void serialize(Archive &archive)
	{
		archive(m_name, m_category, m_handle, m_pins, m_pin_indices, m_variant);
	}

  private:
	std::string m_name;
	std::string m_category;

	size_t m_handle = ~0ull;

	std::map<size_t, MaterialNodePin> m_pins;
	std::map<std::string, size_t>     m_pin_indices;

	Variant m_variant;
};

template <typename _Ty>
class EXPORT_API MaterialNode
{
  public:
	static _Ty &GetInstance()
	{
		static _Ty node;
		return node;
	}

	virtual MaterialNodeDesc Create(size_t &handle) = 0;

	virtual void OnImGui(MaterialNodeDesc &node_desc, Editor *editor) = 0;

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, const MaterialGraphDesc &graph_desc, Renderer *renderer, MaterialCompilationContext *context) = 0;
};
}        // namespace Ilum