#pragma once

#include <rttr/registration.h>

#include "cereal/cereal.hpp"
#include <cereal/types/string.hpp>

#include <glm/glm.hpp>

#include <entt.hpp>

namespace cereal
{
template <class Archive>
void serialize(Archive &archive, glm::vec3 &v)
{
	archive(v.x, v.y, v.z);
}

template <class Archive>
void serialize(Archive &archive, glm::vec4 &v)
{
	archive(v.x, v.y, v.z, v.w);
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
/*
namespace cereal
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
#define ARCHIVE_WRITE_SEQUENTIAL(TYPE)                                       \
    if (type == rttr::type::get<std::vector<TYPE>>())                        \
    {                                                                        \
        std::vector<TYPE> vec = m.convert<std::vector<TYPE>>();              \
        archive(vec.size());                                                 \
        archive(cereal::binary_data(vec.data(), vec.size() * sizeof(TYPE))); \
        finish = true;                                                       \
    }

        ARCHIVE_WRITE_SEQUENTIAL(uint8_t)
        ARCHIVE_WRITE_SEQUENTIAL(uint32_t)

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
    // ARCHIVE_WRITE_SPECIAL_CASES(glm::vec3)
    // ARCHIVE_WRITE_SPECIAL_CASES(glm::vec4)
    // ARCHIVE_WRITE_SPECIAL_CASES(glm::mat4)
    // ARCHIVE_WRITE_SPECIAL_CASES(entt::entity)
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

        // ARCHIVE_READ_SEQUENTIAL(std::vector<uint8_t>)
        // ARCHIVE_READ_SEQUENTIAL(std::vector<uint32_t>)

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
    // ARCHIVE_READ_SPECIAL_CASES(glm::vec3)
    // ARCHIVE_READ_SPECIAL_CASES(glm::vec4)
    // ARCHIVE_READ_SPECIAL_CASES(glm::mat4)
    // ARCHIVE_READ_SPECIAL_CASES(entt::entity)
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
}        // namespace cereal
*/

namespace Ilum
{
template <class Archive>
class Serializer
{
  public:
	explicit Serializer(std::ofstream &os) :
	    m_archive(os)
	{
	}

	~Serializer() = default;

	// template <typename T, cereal::traits::EnableIf<!cereal::traits::is_output_serializable<T, Archive>::value> = cereal::traits::sfinae>
	// void operator()(T &&t)
	//{
	//	rttr::variant var = t;
	//	serialize(var);
	// }

	// template <typename T, cereal::traits::EnableIf<cereal::traits::is_output_serializable<T, Archive>::value> = cereal::traits::sfinae>
	// void operator()(T &&t)
	//{
	//	m_archive(std::forward<T>(t));
	// }

	template <typename T, cereal::traits::EnableIf<cereal::traits::is_output_serializable<T, Archive>::value> = cereal::traits::sfinae>
	void Serialize(T &&t)
	{
		m_archive(std::forward<T>(t));
	}

	template <typename T, cereal::traits::EnableIf<!cereal::traits::is_output_serializable<T, Archive>::value> = cereal::traits::sfinae>
	std::enable_if_t<!std::is_same_v<T, rttr::variant>> Serialize(T &&t)
	{
		rttr::variant var = t;
		Serialize(std::forward<rttr::variant>(var));
	}

	template <typename T, cereal::traits::EnableIf<!cereal::traits::is_output_serializable<T, Archive>::value> = cereal::traits::sfinae>
	std::enable_if_t<std::is_same_v<T, rttr::variant>> Serialize(T &&var)
	{
		auto          type = var.get_type();
		rttr::variant m    = var;
		if (type.is_wrapper())
		{
			m    = m.extract_wrapped_value();
			type = m.get_type();
		}
		std::string name = type.get_name().to_string();

		Serialize(name);

		if (type.is_arithmetic())
		{
#define ARCHIVE_WRITE_ARITHMETIC(TYPE)   \
	if (type == rttr::type::get<TYPE>()) \
	{                                    \
		Serialize(m.convert<TYPE>());    \
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
#define ARCHIVE_WRITE_SEQUENTIAL(TYPE, CONTAINER)                              \
	if (type == rttr::type::get<CONTAINER<TYPE>>())                            \
	{                                                                          \
		CONTAINER<TYPE> vec = m.convert<CONTAINER<TYPE>>();                    \
		Serialize(vec.size());                                                 \
		Serialize(cereal::binary_data(vec.data(), vec.size() * sizeof(TYPE))); \
		finish = true;                                                         \
	}

			ARCHIVE_WRITE_SEQUENTIAL(uint8_t, std::vector)
			ARCHIVE_WRITE_SEQUENTIAL(uint32_t, std::vector)

			if (!finish)
			{
				auto seq_view = m.create_sequential_view();
				Serialize(seq_view.get_size());
				for (auto &elem : seq_view)
				{
					Serialize(elem);
				}
			}
		}
		else if (type.is_associative_container())
		{
			auto ass_view = m.create_associative_view();
			Serialize(ass_view.get_size());
			for (auto &[key, val] : ass_view)
			{
				Serialize(key, val);
			}
		}
#define ARCHIVE_WRITE_SPECIAL_CASES(TYPE)     \
	else if (type == rttr::type::get<TYPE>()) \
	{                                         \
		Serialize(m.convert<TYPE>());         \
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
				Serialize(prop.get_value(m));
			}
		}
		else if (type.is_enumeration())
		{
			auto name = type.get_enumeration().value_to_name(m);
			Serialize(m.convert<uint64_t>());
		}
	}

	template <typename T1, typename... Tn>
	void Serialize(T1 &&t1, Tn &&...tn)
	{
		Serialize(std::forward<T1>(t1));
		Serialize(std::forward<Tn>(tn)...);
	}

	template <typename T1, typename... Tn>
	void operator()(T1 &&t1, Tn &&...tn)
	{
		Serialize(std::forward<T1>(t1), std::forward<Tn>(tn)...);
	}

  private:
	Archive m_archive;
};

template <class Archive>
class Deserializer
{
  public:
	explicit Deserializer(std::ifstream &is) :
	    m_archive(is)
	{
	}

	~Deserializer() = default;

	void DeserializeVariant(rttr::variant& var)
	{
		Deserialize(std::forward<rttr::variant>(var));
	}

	template <typename T, cereal::traits::EnableIf<cereal::traits::is_input_serializable<T, Archive>::value> = cereal::traits::sfinae>
	void Deserialize(T &&t)
	{
		m_archive(std::forward<T>(t));
	}

	template <typename T, cereal::traits::EnableIf<!cereal::traits::is_input_serializable<T, Archive>::value> = cereal::traits::sfinae>
	std::enable_if_t<!std::is_same_v<std::remove_reference_t<T>, rttr::variant>> Deserialize(T &&t)
	{
		std::remove_reference_t<T> tmp = t;
		rttr::variant var = tmp;
		//var               = var.extract_wrapped_value();
		DeserializeVariant(var);
		auto temp = var.convert<std::remove_reference_t<T>>();
		t         = std::move(temp);
	}

	template <typename T, cereal::traits::EnableIf<!cereal::traits::is_input_serializable<T, Archive>::value> = cereal::traits::sfinae>
	std::enable_if_t<std::is_same_v<std::remove_reference_t<T>, rttr::variant>> Deserialize(T &&var)
	{
		std::string name;
		Deserialize(name);

		rttr::variant m = var;

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
		Deserialize(val);                \
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
#define ARCHIVE_READ_SEQUENTIAL(TYPE, CONTAINER)                         \
	if (type == rttr::type::get<CONTAINER<TYPE>>())                      \
	{                                                                    \
		CONTAINER<TYPE> tmp;                                             \
		size_t          size = 0;                                        \
		Deserialize(size);                                               \
		tmp.resize(size);                                                \
		m_archive(cereal::binary_data(tmp.data(), size * sizeof(TYPE))); \
		finish = true;                                                   \
	}

			ARCHIVE_READ_SEQUENTIAL(uint8_t, std::vector)
			ARCHIVE_READ_SEQUENTIAL(uint32_t, std::vector)

			if (!finish)
			{
				auto   seq_view = m.create_sequential_view();
				size_t size     = 0;
				Deserialize(size);
				seq_view.set_size(size);
				for (size_t i = 0; i < size; i++)
				{
					rttr::variant elem = seq_view.get_value_type().create();
					Deserialize(elem);
					seq_view.set_value(i, elem);
				}
			}
		}
		else if (type.is_associative_container())
		{
			auto   ass_view = m.create_associative_view();
			size_t size     = 0;
			Deserialize(size);
			for (size_t i = 0; i < size; i++)
			{
				rttr::variant key = ass_view.get_key_type().create();
				rttr::variant val = ass_view.get_value_type().create();
				Deserialize(key, val);
				ass_view.insert(key, val);
			}
		}
		else if (type == rttr::type::get<std::string>())
		{
			std::string str;
			Deserialize(str);
			m = str;
		}
#define ARCHIVE_READ_SPECIAL_CASES(TYPE)      \
	else if (type == rttr::type::get<TYPE>()) \
	{                                         \
		TYPE t;                               \
		Deserialize(t);                       \
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
				Deserialize(prop_val);
				prop.set_value(m, prop_val);
			}
		}
		else if (type.is_enumeration())
		{
			uint64_t enum_name = 0;
			Deserialize(enum_name);
			std::memcpy(&m, &enum_name, sizeof(uint64_t));
		}

		var = std::move(m);
	}

	template <typename T1, typename... Tn>
	void Deserialize(T1 &&t1, Tn &&...tn)
	{
		Deserialize(std::forward<T1>(t1));
		Deserialize(std::forward<Tn>(tn)...);
	}

	template <typename T1, typename... Tn>
	void operator()(T1 &&t1, Tn &&...tn)
	{
		Deserialize(std::forward<T1>(t1), std::forward<Tn>(tn)...);
	}

  private:
	Archive m_archive;
};
}        // namespace Ilum