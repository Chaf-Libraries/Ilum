#pragma once

#include <string>

namespace Ilum
{
class Panel
{
  public:
	Panel() = default;

	virtual ~Panel() = default;

	virtual void draw(float delta_time) = 0;

	const std::string &name() const
	{
		return m_name;
	}

  public:
	bool active = true;

  protected:
	std::string m_name;
};
}        // namespace Ilum