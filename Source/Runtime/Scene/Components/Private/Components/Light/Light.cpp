#include "Light/Light.hpp"

namespace Ilum
{
namespace Cmpt
{
Light::Light(const std::string &name, Node *node) :
    Component(name, node)
{
}
}        // namespace Cmpt
}        // namespace Ilum