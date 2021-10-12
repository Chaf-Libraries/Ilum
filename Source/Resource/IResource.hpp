#pragma once

#include <typeindex>

namespace Ilum
{
	template<typename T>
	class IResource
	{
	  public:
	    IResource() = default;

		~IResource() = default;

		std::type_index getType() const
		{
		    return typeid(T);
		}

		T* get()
		{
		    return static_cast<T *>(this);
		}
	};
}