#pragma once

#include "Resource.hpp"

namespace Ilum
{
class MaterialGraph;

template <>
class TResource<ResourceType::Material> : public Resource
{
  public:
	explicit TResource(size_t uuid);

	explicit TResource(size_t uuid, const std::string &meta, RHIContext *rhi_context);

	virtual ~TResource() override = default;

	virtual void Load(RHIContext *rhi_context, size_t index) override;

	virtual void Import(RHIContext *rhi_context, const std::string &path) override;

	void Save(const std::string &editor_state);

	MaterialGraph *Get();

	const std::string &GetEditorState() const;

  private:
	std::unique_ptr<MaterialGraph> m_material_graph;

	std::string m_editor_state;

	size_t m_last_update;
};
}        // namespace Ilum