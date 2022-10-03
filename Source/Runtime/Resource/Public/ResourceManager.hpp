#pragma once

#include "Resource.hpp"

namespace Ilum
{
class RHIContext;

class ResourceManager
{
  public:
	ResourceManager(RHIContext *rhi_context);

	~ResourceManager();

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

  private:
	Resource *GetResource(size_t uuid, ResourceType type);

	void EraseResource(size_t uuid, ResourceType type);

  private:
	void Import(const std::string &path, ResourceType type);

	void ImportTexture2D(const std::string &path);

	void ImportModel(const std::string &path);

  private:


  private:
	RHIContext *p_rhi_context = nullptr;

	struct Impl;
	std::unique_ptr<Impl> m_impl = nullptr;
};
}        // namespace Ilum