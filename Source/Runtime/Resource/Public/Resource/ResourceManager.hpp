#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class IResource;

template <typename _Ty>
class Resource;

class EXPORT_API ResourceManager
{
  public:
	ResourceManager(RHIContext *rhi_context);

	~ResourceManager();

	template <typename _Ty>
	Resource<_Ty>* Get(size_t uuid)
	{
		return static_cast<Resource<_Ty> *>(Get(typeid(_Ty), uuid));
	}

	template <typename _Ty>
	bool Has(size_t uuid)
	{
		return Has(Get(typeid(_Ty), uuid));
	}

	template <typename _Ty>
	size_t Index(size_t uuid)
	{
		return Index(typeid(_Ty), uuid);
	}

	template<typename _Ty>
	Resource<_Ty> *Import(size_t uuid)
	{
		return static_cast<Resource<_Ty> *>(Import(typeid(_Ty), uuid));
	}

  private:
	IResource *Get(std::type_index index, size_t uuid);

	bool Has(std::type_index index, size_t uuid);

	size_t Index(std::type_index index, size_t uuid);

	IResource *Import(std::type_index index, size_t uuid);

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum