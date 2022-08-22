#pragma once

#include <string>

namespace Ilum
{
class Editor;

class Widget
{
  public:
	Widget(const std::string &name, Editor *editor) :
	    m_name(name), m_editor(editor)
	{
	}

	virtual ~Widget() = default;

	virtual void Tick() = 0;

	inline bool &GetActive()
	{
		return m_active;
	}

	const std::string& GetName() const
	{
		return m_name;
	}

  protected:
	Editor *m_editor = nullptr;

	std::string m_name;

	bool m_active = true;
};
}        // namespace Ilum