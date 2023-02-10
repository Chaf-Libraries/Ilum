#pragma once

#include "Precompile.hpp"

namespace Ilum
{
class  RenderGraphBlackboard
{
  public:
	RenderGraphBlackboard() = default;

	~RenderGraphBlackboard() = default;

	template <typename _Ty>
	_Ty *Add()
	{
		return std::static_pointer_cast<_Ty>(Add(typeid(_Ty), std::make_shared<_Ty>())).get();
	}

	template <typename _Ty>
	bool Has()
	{
		return Has(typeid(_Ty));
	}

	template <typename _Ty>
	_Ty *Get()
	{
		if (!Has<_Ty>())
		{
			Add<_Ty>();
		}
		return std::static_pointer_cast<_Ty>(Get(typeid(_Ty))).get();
	}

	template <typename _Ty>
	RenderGraphBlackboard &Erase()
	{
		return Erase(typeid(_Ty));
	}

  private:
	std::shared_ptr<void> &Add(std::type_index type, std::shared_ptr<void> &&ptr);

	bool Has(std::type_index type);

	std::shared_ptr<void> &Get(std::type_index type);

	RenderGraphBlackboard &Erase(std::type_index type);

  private:
	std::unordered_map<std::type_index, std::shared_ptr<void>> m_data;
};
}        // namespace Ilum
