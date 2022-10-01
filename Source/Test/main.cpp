#pragma once

#include <map>
#include <vector>

#include <rttr/registration.h>

#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

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
		if (seq_view.get_value_type().is_arithmetic())
		{
		}
		else
		{
			std::vector<rttr::variant> var_vec;
			var_vec.reserve(seq_view.get_size());
			for (auto &elem : seq_view)
			{
				var_vec.push_back(elem);
			}
			archive(var_vec);
		}
	}
	else if (type.is_associative_container())
	{
		auto                       ass_view = m.create_associative_view();
		std::vector<rttr::variant> key_vec;
		std::vector<rttr::variant> val_vec;
		key_vec.reserve(ass_view.get_size());
		val_vec.reserve(ass_view.get_size());
		for (auto &[key, val] : ass_view)
		{
			key_vec.push_back(key);
			val_vec.push_back(val);
		}
		archive(key_vec, val_vec);
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
		std::string name = type.get_enumeration().value_to_name(m).to_string();
		archive(type.get_enumeration().value_to_name(m).to_string());
	}
}

template <class Archive>
void load(Archive &archive, rttr::variant &m)
{
	std::string name;
	archive(name);

	rttr::type type = m.get_type();

	if (type.is_valid())
	{
		type = m.get_type();
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
		auto seq_view = m.create_sequential_view();
		if (seq_view.get_value_type().is_arithmetic())
		{
		}
		else
		{
			std::vector<rttr::variant> var_vec;
			archive(var_vec);
			seq_view.set_size(var_vec.size());
			for (size_t i = 0; i < var_vec.size(); i++)
			{
				seq_view.set_value(i, var_vec[i]);
			}
		}
	}
	else if (type.is_associative_container())
	{
		auto                       ass_view = m.create_associative_view();
		std::vector<rttr::variant> key_vec;
		std::vector<rttr::variant> val_vec;
		archive(key_vec);
		archive(val_vec);
		ass_view.clear();
		for (size_t i = 0; i < key_vec.size(); i++)
		{
			ass_view.insert(key_vec[i], val_vec[i]);
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
struct Test
{
	std::map<int, float>  a;
	std::vector<uint32_t> c;
	rttr::variant         config;
};

RTTR_REGISTRATION
{
	rttr::registration::class_<Test>("Test")
	    .constructor<>()(rttr::policy::ctor::as_object)
	    .property("a", &Test::a)
	    .property("config", &Test::config)
	    .property("c", &Test::c);
}

template <class Archive, typename _Ty, typename = typename std::enable_if_t<!std::is_enum_v<_Ty>>>
inline void serialize(Archive &archive, _Ty &t)
{
	rttr::variant var = t;
	archive(var);
	t = var.convert<_Ty>();
}

}        // namespace Ilum

#include <cereal/archives/binary.hpp>
#include <fstream>

using InputArchive  = cereal::BinaryInputArchive;
using OutputArchive = cereal::BinaryOutputArchive;

int main()
{
	Ilum::Test test = {};
	test.a          = {{1, 0.f},
              {2, 0.f},
              {43, 100.f}};
	test.c.resize(30000000);

	Ilum::Test config = {};
	config.a          = {{1, 0.f},
                {2, 0.f},
                {43, 100.f}};
	config.c.resize(5);

	test.config = config;

	Ilum::Test test2 = {};

	{
		std::ofstream os("test.json");
		OutputArchive archive(os);
		Ilum::serialize(archive, test);
	}

	{
		std::ifstream is("test.json");
		InputArchive  archive(is);
		Ilum::serialize(archive, test2);
		Ilum::Test cfg = test2.config.convert<Ilum::Test>();
	}
	return 0;
}