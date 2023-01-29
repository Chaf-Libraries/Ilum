#pragma once

#include <Scene/Component.hpp>

namespace Ilum
{
namespace Cmpt
{
class Camera;

class Light : public Component
{
  public:
	Light(const char *name, Node *node);

	virtual ~Light() = default;

	virtual void OnImGui() = 0;

	virtual std::type_index GetType() const = 0;

	virtual void Save(OutputArchive &archive) const = 0;

	virtual void Load(InputArchive &archive) = 0;

	// GPU data size
	virtual size_t GetDataSize() const = 0;

	// GPU data info
	virtual void *GetData(Camera *camera = nullptr) = 0;

  private:
	// Shadow
	bool  m_shadow_enabled = true;
	float m_shadow_bias    = 0.1f;
};
}        // namespace Cmpt
}        // namespace Ilum