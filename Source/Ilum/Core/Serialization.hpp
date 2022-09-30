//#pragma once
//
//#include "Precompile.hpp"
//#include "Singleton.hpp"
//
//#include <rttr/registration.h>
//
//#include <cereal/details/traits.hpp>
//#include <cereal/types/string.hpp>
//#include <cereal/types/vector.hpp>
//
//#include <glm/glm.hpp>
//
//namespace cereal
//{
//template <class Archive>
//class TSerializer : public Ilum::Singleton<TSerializer<Archive>>
//{
//  public:
//	using SerializeFunction = std::function<void(Archive &ar, rttr::variant &var, const rttr::property &)>;
//
//  public:
//	template <typename _Ty>
//	void Serialize(Archive &ar, _Ty &var)
//	{
//		ar(var);
//	}
//
//	void Serialize(Archive &ar, rttr::variant &var)
//	{
//		if (!var.is_valid())
//		{
//			std::string type_name;
//			ar(type_name);
//			rttr::constructor ctor = rttr::type::get_by_name(type_name).get_constructor();        // 2nd way with the constructor class
//			var                    = ctor.invoke();
//		}
//		else
//		{
//			ar(std::string(var.get_type().get_name()));
//		}
//
//		auto type = var.get_type();
//		for (auto &prop : type.get_properties())
//		{
//			auto prop_var = prop.get_value(var);
//			if (prop.get_type() == rttr::type::get<rttr::variant>())
//			{
//				// Some component might by rttr::variant type
//				if (prop_var.get_type().get_name().empty())
//				{
//					std::string prop_type_name = "";
//					ar(prop_type_name);
//					if (prop_type_name.empty())
//					{
//						// Empty variant
//						continue;
//					}
//					prop_var = rttr::type::get_by_name(prop_type_name).create();
//				}
//				else
//				{
//					ar(prop_var.get_type().get_name().to_string());
//				}
//			}
//			m_func[prop_var.get_type()](ar, var, prop);
//		}
//	}
//
//	template <typename _Ty>
//	inline void SerializeProperty(Archive &ar, rttr::variant &var, const rttr::property &prop)
//	{
//		rttr::variant prop_var = prop.get_value(var);
//
//		_Ty raw_val = prop_var.convert<_Ty>();
//
//		Serialize(ar, raw_val);
//		prop_var = raw_val;
//		prop.set_value(var, prop_var);
//	}
//
//	template <typename _Ty>
//	void RegisterType()
//	{
//		m_func[rttr::type::get<_Ty>()] = [this](Archive &ar, rttr::variant &var, const rttr::property &prop) { SerializeProperty<_Ty>(ar, var, prop); };
//	}
//
//	template <>
//	void RegisterType<rttr::variant>()
//	{
//		m_func[rttr::type::get<rttr::variant>()] = [this](Archive &ar, rttr::variant &var, const rttr::property &prop) {
//			auto prop_var = prop.get_value(var);
//			Serialize(ar, prop_var);
//			auto type = prop_var.get_type();
//			prop.set_value(var, prop_var);
//		};
//	}
//
//  private:
//	std::unordered_map<rttr::type, SerializeFunction> m_func;
//};
//
//template <class Archive>
//void serialize(Archive &archive, rttr::variant &var)
//{
//	TSerializer<Archive>::GetInstance().Serialize(archive, var);
//}
//
//template <typename Archive, size_t N, typename _Ty>
//void serialize(Archive &archive, glm::vec<N, _Ty> &var)
//{
//	for (uint32_t i = 0; i < N; i++)
//	{
//		archive(var[i]);
//	}
//}
//
//template<typename Archive, size_t C, size_t R, typename _Ty>
//void serialize(Archive& archive, glm::mat<C, R, _Ty>& var)
//{
//	for (uint32_t i = 0; i < C; i++)
//	{
//		for (uint32_t j = 0; j < R; j++)
//		{
//			archive(var[i][j]);
//		}
//	}
//}
//}        // namespace cereal