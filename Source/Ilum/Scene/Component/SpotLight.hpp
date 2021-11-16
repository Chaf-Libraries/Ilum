#pragma once

#include "Light.hpp"

namespace Ilum::cmpt
{
struct SpotLight : public TLight<LightType::Spot>
{
};
}