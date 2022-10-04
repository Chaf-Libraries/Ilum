#pragma once

#include <rttr/registration.h>

#include <glm/glm.hpp>

#include <entt.hpp>

namespace cereal
{
template <class Archive, typename T, int N>
void serialize(Archive &archive, glm::vec<N, T> &v)
{
	for (int i = 0; i < N; i++)
	{
		archive(v[i]);
	}
}

template <class Archive>
void serialize(Archive &archive, glm::mat4 &m)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			archive(m[i][j]);
		}
	}
}

}        // namespace cereal

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
#define ARCHIVE_WRITE_SPECIAL_CASES(TYPE)     \
	else if (type == rttr::type::get<TYPE>()) \
	{                                         \
		archive(m.convert<TYPE>());           \
	}
	ARCHIVE_WRITE_SPECIAL_CASES(std::string)
	ARCHIVE_WRITE_SPECIAL_CASES(glm::vec3)
	ARCHIVE_WRITE_SPECIAL_CASES(glm::vec4)
	ARCHIVE_WRITE_SPECIAL_CASES(glm::mat4)
	ARCHIVE_WRITE_SPECIAL_CASES(entt::entity)
	else if (type.is_class())
	{
		for (auto &prop : type.get_properties())
		{
			archive(prop.get_value(m));
		}
	}
	else if (type.is_enumeration())
	{
		auto name = type.get_enumeration().value_to_name(m);
		archive(m.convert<uint64_t>());
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
		auto   ass_view = m.create_associative_view();
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
#define ARCHIVE_READ_SPECIAL_CASES(TYPE)      \
	else if (type == rttr::type::get<TYPE>()) \
	{                                         \
		TYPE t;                               \
		archive(t);                           \
		m = t;                                \
	}
	ARCHIVE_READ_SPECIAL_CASES(std::string)
	ARCHIVE_READ_SPECIAL_CASES(glm::vec3)
	ARCHIVE_READ_SPECIAL_CASES(glm::vec4)
	ARCHIVE_READ_SPECIAL_CASES(glm::mat4)
	ARCHIVE_READ_SPECIAL_CASES(entt::entity)
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
		uint64_t enum_name = 0;
		archive(enum_name);
		std::memcpy(&m, &enum_name, sizeof(uint64_t));
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