#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <rttr/registration.h>

namespace Ilum
{
template <class T>
inline void HashCombine(size_t &seed, const T &v)
{
	std::hash<T> hasher;
	glm::detail::hash_combine(seed, hasher(v));
}

template <typename T>
inline void HashCombine(size_t &seed, const std::vector<T> &v)
{
	for (auto& data : v)
	{
		HashCombine(seed, data);
	}
}

template <class T1, class... T2>
inline void HashCombine(size_t &seed, const T1 &v1, const T2 &...v2)
{
	HashCombine(seed, v1);
	HashCombine(seed, v2...);
}

template<typename T>
inline size_t Hash(const T& v)
{
	size_t hash = 0;
	HashCombine(hash, v);
	return hash;
}

template <typename T1, typename... T2>
inline size_t Hash(const T1 &v1, const T2 &...v2)
{
	size_t hash = 0;
	HashCombine(hash, v1, v2...);
	return hash;
}
}        // namespace Ilum

namespace std
{
template <>
struct hash<rttr::variant>
{
	size_t operator()(const rttr::variant &var) const
	{
		auto          type = var.get_type();
		rttr::variant m    = var;
		if (type.is_wrapper())
		{
			m    = m.extract_wrapped_value();
			type = m.get_type();
		}

		if (type.is_arithmetic())
		{
#define HASH_ARITHMETIC(TYPE)\
			if (type == rttr::type::get<TYPE>())\
			{\
				return Ilum::Hash(m.convert<TYPE>());\
			}
			HASH_ARITHMETIC(bool)
			HASH_ARITHMETIC(char)
			HASH_ARITHMETIC(float)
			HASH_ARITHMETIC(double)
			HASH_ARITHMETIC(uint8_t)
			HASH_ARITHMETIC(uint16_t)
			HASH_ARITHMETIC(uint32_t)
			HASH_ARITHMETIC(uint64_t)
			HASH_ARITHMETIC(int8_t)
			HASH_ARITHMETIC(int16_t)
			HASH_ARITHMETIC(int32_t)
			HASH_ARITHMETIC(int64_t)
		}
		else if (type.is_sequential_container())
		{
			auto seq_view = m.create_sequential_view();
			size_t hash_val = 0;
			for (auto &elem : seq_view)
			{
				Ilum::HashCombine(hash_val, elem);
			}
			return hash_val;
		}
		else if (type.is_associative_container())
		{
			auto ass_view = m.create_associative_view();
			size_t hash_val = 0;
			for (auto& [key, val] : ass_view)
			{
				Ilum::HashCombine(hash_val, key, val);
			}
			return hash_val;
		}
		else if (type == rttr::type::get<std::string>())
		{
			return Ilum::Hash(m.to_string());
		}
		else if (type.is_enumeration())
		{
			return Ilum::Hash(type.get_enumeration().value_to_name(m).to_string());
		}
		else if (type.is_class())
		{
			size_t hash_val = 0;
			for (auto &prop : type.get_properties())
			{
				hash_val = Ilum::Hash(prop.get_value(m));
			}
			return hash_val;
		}
		return 0;
	}
};
}