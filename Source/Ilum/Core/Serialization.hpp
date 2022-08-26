#pragma once

#include "Singleton.hpp"

#include <rttr/registration.h>

namespace Ilum
{
template <typename Archieve, typename _Ty>
inline static void serialize_property(Archieve &ar, const rttr::variant &var, const rttr::property &prop)
{
	rttr::variant val = prop.get_value(var);

	_Ty raw_val = val.convert<_Ty>();

	ar(raw_val);

	prop.set_value(var, raw_val);
}

template <typename Archieve>
class TSerializer : public Singleton<TSerializer<Archieve>>
{
  public:
	template <typename _Ty>
	static void RegisterType()
	{
		auto t = rttr::type::get<decltype(_Ty)>();
		SerialFunctions.emplace(rttr::type::get<_Ty>(), [](Archieve &ar, const rttr::variant &var, const rttr::property &prop) { serialize_property<Archieve, _Ty>(ar, var, prop); });
	}

	inline void serialize(Archieve &ar, const rttr::variant &var)
	{
		for (auto &property_ : var.get_type().get_properties())
		{
			if (!property_.get_type().get_properties().empty())
			{
				serialize(ar, property_.get_value(var));
			}
			else
			{
				SerialFunctions[property_.get_type()](ar, var, property_);
			}
		}
	}

  public:
	inline static std::unordered_map<rttr::type, std::function<void(Archieve &, const rttr::variant &, const rttr::property &)>> SerialFunctions;
};

}        // namespace Ilum