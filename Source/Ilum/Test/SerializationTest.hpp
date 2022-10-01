#pragma once
#if defined(__REFLECTION_PARSER__)
#	define META(...) __attribute__((annotate(#    __VA_ARGS__)))
#	define CLASS(class_name, ...) class __attribute__((annotate(#    __VA_ARGS__))) class_name
#	define STRUCT(struct_name, ...) struct __attribute__((annotate(#    __VA_ARGS__))) struct_name
#	define ENUM(enum_name, ...) enum class __attribute__((annotate(#    __VA_ARGS__))) enum_name
#else
#	define META(...)
#	define CLASS(class_name, ...) class class_name
#	define STRUCT(struct_name, ...) struct struct_name
#	define ENUM(enum_name, ...) enum class enum_name
#endif        // __REFLECTION_PARSER__
#include <cereal/types/string.hpp>
#include <rttr/registration.h>

STRUCT(TestStruct, Reflection, Serialization)
{
	META(Reflection)
	int a;

	META(Reflection)
	float b;

	META(Reflection)
	std::vector<std::string> v;

	META(Reflection)
	std::map<std::string, int> m;

	TestStruct()
	{
	}
};

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
#include <cassert>
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
			m = m.extract_wrapped_value();
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
}        // namespace rttr

template<class Archive, typename _Ty>
void serialize(Archive& archive, _Ty& t)
{
	rttr::variant var = t;
	archive(var);
	t = var.convert<_Ty>();
}