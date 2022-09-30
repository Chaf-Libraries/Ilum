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

STRUCT(TestStruct, Reflection, Serialization)
{
	META(Reflection)
	int a;

	META(Reflection)
	float b;

	TestStruct()
	{

	}
};

#include <rttr/registration.h>

namespace cereal
{
template <class Archive>
inline void serialize(Archive& archive, rttr::variant& m)
{
	size_t               size = m.get_type().get_sizeof();
	std::string          name = m.get_type().get_name().data();
	std::vector<uint8_t> data(size);
	std::memcpy(data.data(), &m, size);

	archive(size, name, data);

	auto type = rttr::type::get_by_name(name);
	
	m = type.create();
	      size = m.get_type().get_sizeof();
	 name = m.get_type().get_name().data();
	std::memcpy(&m, data.data(), size);
}
}        // namespace cereal