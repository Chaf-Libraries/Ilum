#pragma once

#include "Utils/PCH.hpp"

#include <glm/glm.hpp>

namespace Ilum
{
class AABB;

class Bound
{
  public:
	enum class Type
	{
		AABB
	};

	Bound();

	~Bound() = default;

	Bound *get() const;

	virtual bool valid() const = 0;

	virtual void add(const glm::vec3 &point) = 0;

	virtual void add(const std::vector<glm::vec3> &points, const std::vector<uint32_t> &indices) = 0;

	virtual void transform(const glm::mat4 &matrix) = 0;

	void set(Type type);

  public:
  private:
	Type m_type = Type::AABB;
	scope<Bound> m_ptr  = nullptr;
};
}        // namespace Ilum