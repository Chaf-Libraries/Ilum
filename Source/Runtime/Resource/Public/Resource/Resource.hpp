#pragma once

#include <Core/Core.hpp>

namespace Ilum
{
class RHITexture;
class RHIContext;

enum class ResourceType
{
	Unknown,
	Prefab,
	Mesh,
	SkinnedMesh,
	Texture2D,
	Animation,
	Material,
	RenderPipeline,
	Scene,
};

class EXPORT_API IResource
{
  public:
	explicit IResource(const std::string &name);

	explicit IResource(RHIContext *rhi_context, const std::string &name, ResourceType type);

	virtual ~IResource() = default;

	const std::string &GetName() const;

	virtual bool Validate() const
	{
		return true;
	}

	virtual void Load(RHIContext *rhi_context)
	{
	}

	size_t GetUUID() const;

	RHITexture *GetThumbnail() const;

  protected:
	void UpdateThumbnail(RHIContext *rhi_context, const std::vector<uint8_t> &thumbnail_data);

  protected:
	std::string m_name;

	std::unique_ptr<RHITexture> m_thumbnail = nullptr;
};

template <ResourceType Type>
class EXPORT_API Resource : public IResource
{
  public:
	Resource() = default;

	virtual ~Resource() override = default;
};
}        // namespace Ilum
