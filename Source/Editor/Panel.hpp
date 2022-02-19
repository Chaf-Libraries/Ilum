#pragma once

#include <string>

namespace Ilum::Editor
{
class Panel
{
  public:
	Panel()  = default;
	virtual ~Panel() = default;

	virtual void Show() = 0;

	inline const std::string &GetName() const
	{
		return m_name;
	}

  public:
	bool active = true;

  protected:
	std::string m_name;
};
}        // namespace Ilum::Editor