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

#pragma once

#include <rttr/registration.h>

namespace rttr
{
template <class Archive>
void save(Archive &archive, rttr::variant const &var)
{
	auto          type = var.get_type();
	rttr::variant m    = var;
	if (type.is_wrapper())
	{
		m    = m.extract_wrapped_value();
		type = m.get_type();
	}
	std::string name = type.get_name().to_string();

	archive(name);

	if (type.is_arithmetic())
	{
#define ARCHIVE_WRITE_ARITHMETIC(TYPE)   \
	if (type == rttr::type::get<TYPE>()) \
	{                                    \
		archive(m.convert<TYPE>());      \
	}
		ARCHIVE_WRITE_ARITHMETIC(bool)
		ARCHIVE_WRITE_ARITHMETIC(char)
		ARCHIVE_WRITE_ARITHMETIC(float)
		ARCHIVE_WRITE_ARITHMETIC(double)
		ARCHIVE_WRITE_ARITHMETIC(uint8_t)
		ARCHIVE_WRITE_ARITHMETIC(uint16_t)
		ARCHIVE_WRITE_ARITHMETIC(uint32_t)
		ARCHIVE_WRITE_ARITHMETIC(uint64_t)
		ARCHIVE_WRITE_ARITHMETIC(int8_t)
		ARCHIVE_WRITE_ARITHMETIC(int16_t)
		ARCHIVE_WRITE_ARITHMETIC(int32_t)
		ARCHIVE_WRITE_ARITHMETIC(int64_t)
	}
	else if (type.is_sequential_container())
	{
		auto seq_view = m.create_sequential_view();
		archive(seq_view.get_size());
		for (auto &elem : seq_view)
		{
			archive(elem);
		}
	}
	else if (type.is_associative_container())
	{
		auto ass_view = m.create_associative_view();
		archive(ass_view.get_size());
		for (auto &[key, val] : ass_view)
		{
			archive(key, val);
		}
	}
	else if (type == rttr::type::get<std::string>())
	{
		archive(m.to_string());
	}
	else if (type.is_class())
	{
		for (auto &prop : type.get_properties())
		{
			archive(prop.get_value(m));
		}
	}
}

template <class Archive>
void load(Archive &archive, rttr::variant &m)
{
	std::string name;
	archive(name);

	rttr::type type = rttr::type::get_by_name(name);

	if (!m.is_valid())
	{
		m = type.create();
	}
	else
	{
		type = m.get_type();
		if (type.is_wrapper())
		{
			m    = m.extract_wrapped_value();
			type = m.get_type();
		}
	}

	bool a = m.is_valid();
	bool b = type.is_arithmetic();

	if (type.is_arithmetic())
	{
#define ARCHIVE_READ_ARITHMETIC(TYPE)    \
	if (type == rttr::type::get<TYPE>()) \
	{                                    \
		TYPE val = (TYPE) 0;             \
		archive(val);                    \
		m = val;                         \
	}
		ARCHIVE_READ_ARITHMETIC(bool)
		ARCHIVE_READ_ARITHMETIC(char)
		ARCHIVE_READ_ARITHMETIC(float)
		ARCHIVE_READ_ARITHMETIC(double)
		ARCHIVE_READ_ARITHMETIC(uint8_t)
		ARCHIVE_READ_ARITHMETIC(uint16_t)
		ARCHIVE_READ_ARITHMETIC(uint32_t)
		ARCHIVE_READ_ARITHMETIC(uint64_t)
		ARCHIVE_READ_ARITHMETIC(int8_t)
		ARCHIVE_READ_ARITHMETIC(int16_t)
		ARCHIVE_READ_ARITHMETIC(int32_t)
		ARCHIVE_READ_ARITHMETIC(int64_t)
	}
	else if (type.is_sequential_container())
	{
		auto   seq_view = m.create_sequential_view();
		size_t size     = 0;
		archive(size);
		seq_view.set_size(size);
		for (size_t i = 0; i < size; i++)
		{
			rttr::variant elem = seq_view.get_value(i);
			archive(elem);
			seq_view.set_value(i, elem);
		}
	}
	else if (type.is_associative_container())
	{
		auto   ass_view = m.create_associative_view();
		size_t size     = 0;
		archive(size);
		for (size_t i = 0; i < size; i++)
		{
			rttr::variant key, val;
			archive(key, val);
			ass_view.insert(key, val);
		}
	}
	else if (type == rttr::type::get<std::string>())
	{
		std::string str;
		archive(str);
		m = str;
	}
	else if (type.is_class())
	{
		for (auto &prop : type.get_properties())
		{
			rttr::variant prop_val = prop.get_value(m);
			archive(prop_val);
			prop.set_value(m, prop_val);
		}
	}
}
}