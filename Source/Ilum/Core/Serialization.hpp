#pragma once

#include "Singleton.hpp"

#include <rttr/registration.h>

#include <cereal/details/traits.hpp>

namespace Ilum
{
template <typename Archive>
class TSerializer : public Singleton<TSerializer<Archive>>
{
  public:
	template <typename _Ty>
	void Serialize(Archive &ar, _Ty &var)
	{
		ar(var);
	}

	void Serialize(Archive &ar, rttr::variant &var)
	{
		auto type = var.get_type();
		for (auto &prop : type.get_properties())
		{
			auto prop_var  = prop.get_value(var);
			if (prop.get_type() == rttr::type::get<rttr::variant>())
			{
				// Some component might by rttr::variant type
				if (prop_var.get_type().get_name().empty())
				{
					std::string prop_type_name = "";
					ar(prop_type_name);
					if (prop_type_name.empty())
					{
						// Empty variant
						continue;
					}
					prop_var = rttr::type::get_by_name(prop_type_name).create();
				}
				else
				{
					ar(prop_var.get_type().get_name().to_string());
				}
			}

			TSerializer<Archive>::GetInstance().SerialFunctions[prop_var.get_type()](ar, var, prop);
		}
	}

	template <typename _Ty>
	inline void SerializeProperty(Archive &ar, rttr::variant &var, const rttr::property &prop)
	{
		rttr::variant val = prop.get_value(var);

		_Ty raw_val = val.convert<_Ty>();

		Serialize(ar, raw_val);

		prop.set_value(var, raw_val);
	}

	template <>
	void SerializeProperty<rttr::variant>(Archive &ar, rttr::variant &var, const rttr::property &prop)
	{
		auto prop_var = prop.get_value(var);
		if (prop_var.is_valid())
		{
			serialize(ar, prop_var);
			prop.set_value(var, prop_var);
		}
	}

  public:
	template <typename _Ty>
	void RegisterType()
	{
		SerialFunctions[rttr::type::get<_Ty>()] = [this](Archive &ar, rttr::variant &var, const rttr::property &prop) { SerializeProperty<_Ty>(ar, var, prop); };
	}

	template <>
	void RegisterType<rttr::variant>()
	{
		SerialFunctions[rttr::type::get<rttr::variant>()] = [this](Archive &ar, rttr::variant &var, const rttr::property &prop) {
			auto prop_var = prop.get_value(var);
			Serialize(ar, prop_var);
			prop.set_value(var, prop_var);
		};
	}

  public:
	std::unordered_map<rttr::type, std::function<void(Archive &, rttr::variant &, const rttr::property &)>> SerialFunctions;
};

template <typename Archive>
void serialize(Archive &ar, rttr::variant &var)
{
	TSerializer<Archive>::GetInstance().Serialize(ar, var);
}

}        // namespace Ilum