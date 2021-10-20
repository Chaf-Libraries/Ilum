#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Subsystem.hpp"

#include "Panel.hpp"

namespace Ilum
{
class Editor : public TSubsystem<Editor>
{
  public:
	Editor(Context *context);

	~Editor() = default;

	virtual bool onInitialize() override;

	virtual void onPreTick() override;

	virtual void onTick(float delta_time) override;

	virtual void onPostTick() override;

	virtual void onShutdown() override;

  private:
	std::vector<scope<Panel>> m_panels;
};
}        // namespace Ilum