#pragma once

#include "Resource.hpp"

namespace Ilum
{
class RHIContext;
class RHITexture;

class ResourceManager
{
  public:
	ResourceManager(RHIContext *rhi_context);

	~ResourceManager();

	void Tick();

	bool IsUpdate();

	template <ResourceType _Ty>
	void Import(const std::string &path)
	{
		Import(path, _Ty);
	}

	template <ResourceType _Ty>
	TResource<_Ty> *GetResource(size_t uuid)
	{
		auto *resource = GetResource(uuid, _Ty);
		return resource ? resource->CastTo<_Ty>() : nullptr;
	}

	template <ResourceType _Ty>
	void EraseResource(size_t uuid)
	{
		EraseResource(uuid, _Ty);
	}

	template <ResourceType _Ty>
	RHITexture *GetThumbnail()
	{
		return GetThumbnail(_Ty);
	}

	template <ResourceType _Ty>
	size_t GetResourceIndex(size_t uuid)
	{
		return GetResourceIndex(uuid, _Ty);
	}

	template <ResourceType _Ty>
	const std::string &GetResourceMeta(size_t uuid)
	{
		return GetResourceMeta(uuid, _Ty);
	}

	template <ResourceType _Ty>
	bool IsUpdate()
	{
		return IsUpdate(_Ty);
	}

	template <ResourceType _Ty>
	bool IsValid(size_t uuid)
	{
		return IsValid(uuid, _Ty);
	}

	template <ResourceType _Ty>
	const std::vector<size_t> &GetResourceUUID() const
	{
		return GetResourceUUID(_Ty);
	}

	template <ResourceType _Ty>
	const std::vector<size_t> &GetResourceValidUUID() const
	{
		return GetResourceValidUUID(_Ty);
	}

  private:
	Resource *GetResource(size_t uuid, ResourceType type);

	void EraseResource(size_t uuid, ResourceType type);

	void Import(const std::string &path, ResourceType type);

	RHITexture *GetThumbnail(ResourceType type);

	size_t GetResourceIndex(size_t uuid, ResourceType type);

	const std::string &GetResourceMeta(size_t uuid, ResourceType type);

	bool IsUpdate(ResourceType type);

	bool IsValid(size_t uuid, ResourceType type);

	const std::vector<size_t> &GetResourceUUID(ResourceType type) const;

	const std::vector<size_t> &GetResourceValidUUID(ResourceType type) const;

  private:
	void ScanLocalMeta();

  private:
	RHIContext *p_rhi_context = nullptr;

	struct Impl;
	std::unique_ptr<Impl> m_impl = nullptr;
};
}        // namespace Ilum