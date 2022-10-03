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
		bool finish = false;
#define ARCHIVE_WRITE_SEQUENTIAL(TYPE)   \
	if (type == rttr::type::get<TYPE>()) \
	{                                    \
		archive(m.convert<TYPE>());      \
		finish = true;                   \
	}

		ARCHIVE_WRITE_SEQUENTIAL(std::vector<uint8_t>)
		ARCHIVE_WRITE_SEQUENTIAL(std::vector<uint32_t>)

		if (!finish)
		{
			auto seq_view = m.create_sequential_view();
			archive(seq_view.get_size());
			for (auto &elem : seq_view)
			{
				archive(elem);
			}
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
	else if (type.is_enumeration())
	{
		archive(type.get_enumeration().value_to_name(m).to_string());
	}
}

template <class Archive>
void load(Archive &archive, rttr::variant &m)
{
	std::string name;
	archive(name);

	rttr::type type = m.get_type();

	if (!type.is_valid())
	{
		type = rttr::type::get_by_name(name);
	}

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
		bool finish = false;
#define ARCHIVE_READ_SEQUENTIAL(TYPE)    \
	if (type == rttr::type::get<TYPE>()) \
	{                                    \
		TYPE var;                        \
		archive(var);                    \
		m      = var;                    \
		finish = true;                   \
	}

		ARCHIVE_READ_SEQUENTIAL(std::vector<uint8_t>)
		ARCHIVE_READ_SEQUENTIAL(std::vector<uint32_t>)

		if (!finish)
		{
			auto   seq_view = m.create_sequential_view();
			size_t size     = 0;
			archive(size);
			seq_view.set_size(size);
			for (size_t i = 0; i < size; i++)
			{
				rttr::variant elem = seq_view.get_value_type().create();
				archive(elem);
				seq_view.set_value(i, elem);
			}
		}
	}
	else if (type.is_associative_container())
	{
		auto ass_view = m.create_associative_view();
		size_t size     = 0;
		archive(size);
		for (size_t i = 0; i < size; i++)
		{
			rttr::variant key = ass_view.get_key_type().create();
			rttr::variant val = ass_view.get_value_type().create();
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
	else if (type.is_enumeration())
	{
		std::string enum_name;
		archive(enum_name);
		m = type.get_enumeration().name_to_value(enum_name);
	}
}
}        // namespace rttr

namespace Ilum
{
template <class Archive, typename _Ty, typename = typename std::enable_if_t<!std::is_enum_v<_Ty>>>
inline void serialize(Archive &archive, _Ty &t)
{
	rttr::variant var = t;
	archive(var);
	t = var.convert<_Ty>();
}

}        // namespace Ilum