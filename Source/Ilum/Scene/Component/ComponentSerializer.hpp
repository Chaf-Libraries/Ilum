#pragma once

#include <cereal/cereal.hpp>

#include <glm/glm.hpp>

namespace glm
{
template <class Archive>
void serialize(Archive &ar, glm::vec2 &v)
{
	ar(cereal::make_nvp("x", v.x), cereal::make_nvp("y", v.y));
}

template <class Archive>
void serialize(Archive &ar, glm::vec3 &v)
{
	ar(cereal::make_nvp("x", v.x), cereal::make_nvp("y", v.y), cereal::make_nvp("z", v.z));
}

template <class Archive>
void serialize(Archive &ar, glm::vec4 &v)
{
	ar(cereal::make_nvp("x", v.x), cereal::make_nvp("y", v.y), cereal::make_nvp("z", v.z), cereal::make_nvp("w", v.w));
}

template <class Archive>
void serialize(Archive &ar, glm::mat2 &v)
{
	ar(cereal::make_nvp("[0][0]", v[0][0]), cereal::make_nvp("[0][1]", v[0][1]),
	   cereal::make_nvp("[1][0]", v[1][0]), cereal::make_nvp("[1][1]", v[1][1]));
}

template <class Archive>
void serialize(Archive &ar, glm::mat3 &v)
{
	ar(cereal::make_nvp("[0][0]", v[0][0]), cereal::make_nvp("[0][1]", v[0][1]), cereal::make_nvp("[0][2]", v[0][2]),
	   cereal::make_nvp("[1][0]", v[0][0]), cereal::make_nvp("[1][1]", v[0][1]), cereal::make_nvp("[1][2]", v[0][2]),
	   cereal::make_nvp("[2][0]", v[0][0]), cereal::make_nvp("[2][1]", v[0][1]), cereal::make_nvp("[2][2]", v[0][2]));
}

template <class Archive>
void serialize(Archive &ar, glm::mat4 &v)
{
	ar(cereal::make_nvp("[0][0]", v[0][0]), cereal::make_nvp("[0][1]", v[0][1]), cereal::make_nvp("[0][2]", v[0][2]), cereal::make_nvp("[0][3]", v[0][3]),
	   cereal::make_nvp("[1][0]", v[1][0]), cereal::make_nvp("[1][1]", v[1][1]), cereal::make_nvp("[1][2]", v[1][2]), cereal::make_nvp("[1][3]", v[1][3]),
	   cereal::make_nvp("[2][0]", v[2][0]), cereal::make_nvp("[2][1]", v[2][1]), cereal::make_nvp("[2][2]", v[2][2]), cereal::make_nvp("[2][3]", v[2][3]),
	   cereal::make_nvp("[3][0]", v[3][0]), cereal::make_nvp("[3][1]", v[3][1]), cereal::make_nvp("[3][2]", v[3][2]), cereal::make_nvp("[3][3]", v[3][3]), );
}

template <class Archive, typename T1, typename T2, typename... Tn>
void serialize(Archive &ar, T1 &t1, T2 &t2, Tn... tn)
{
	serialize(ar, t1);
	serialize(ar, t2, tn...);
}
}        // namespace glm