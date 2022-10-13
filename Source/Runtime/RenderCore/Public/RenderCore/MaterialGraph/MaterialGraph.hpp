#pragma once

#include "MaterialNode.hpp"

namespace Ilum
{
class RHIContext;
class RHIBuffer;
class RHIDescriptor;

STRUCT(MaterialGraphDesc, Enable)
{
	std::map<size_t, MaterialNodeDesc> nodes;

	std::map<size_t, size_t> links;        // Target - Source

	std::map<size_t, size_t> node_query;

	void AddNode(size_t & handle, MaterialNodeDesc && desc);

	void EraseNode(size_t handle);

	void EraseLink(size_t source, size_t target);

	void Link(size_t source, size_t target);

	bool HasLink(size_t target);

	size_t LinkFrom(size_t target_pin);

	const MaterialNodeDesc &GetNode(size_t pin);

	std::string GetEmitResult(const MaterialNodeDesc &desc, const std::string &pin_name, MaterialEmitInfo& emit_info);

	std::string GetEmitExpression(const MaterialNodeDesc &desc, const std::string &pin_name, MaterialEmitInfo &emit_info);

};

class MaterialGraph
{
	friend class MaterialGraphBuilder;

  public:
	MaterialGraph(RHIContext *rhi_context);

	~MaterialGraph() = default;

  private:
	RHIContext *p_rhi_context = nullptr;
};
}        // namespace Ilum