#pragma once

#include "MaterialNode.hpp"

namespace Ilum
{
class RHIContext;
class RHIBuffer;
class RHIDescriptor;

STRUCT(MaterialGraphDesc, Enable)
{
	std::string name;

	std::map<size_t, MaterialNodeDesc> nodes;

	std::map<size_t, size_t> links;        // Target - Source

	std::map<size_t, size_t> node_query;

	void AddNode(size_t & handle, MaterialNodeDesc && desc);

	void EraseNode(size_t handle);

	void EraseLink(size_t source, size_t target);

	void Link(size_t source, size_t target);

	void UpdateNode(size_t node);

	bool HasLink(size_t target) const;

	size_t LinkFrom(size_t target_pin) const;

	const MaterialNodeDesc &GetNode(size_t node_handle) const;
};

class MaterialGraph
{
	friend class MaterialGraphBuilder;

  public:
	MaterialGraph(RHIContext *rhi_context, const MaterialGraphDesc &desc);

	~MaterialGraph() = default;

	MaterialGraphDesc &GetDesc();

	void EmitShader(size_t pin, ShaderEmitContext &context);

	void Validate(size_t pin, ShaderValidateContext &context);

	void SetUpdate(bool update);

	bool Update();

  private:
	RHIContext *p_rhi_context = nullptr;

	MaterialGraphDesc m_desc;

	bool m_update = false;
};
}        // namespace Ilum