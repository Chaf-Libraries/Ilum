#pragma once

namespace Ilum
{
class Panel
{
  public:
	Panel() = default;

	~Panel() = default;

	virtual void draw() = 0;

  protected:
	bool m_active = true;
};
}        // namespace Ilum