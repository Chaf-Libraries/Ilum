#pragma once

#include <string>

#include <glm/glm.hpp>

namespace Ilum
{
class Serializer
{
  public:
	Serializer() = default;

	~Serializer() = default;

	virtual void serialize(const std::string &file_path) = 0;

	virtual void deserialize(const std::string &file_path) = 0;
};
}        // namespace Ilum