#pragma once

#include "Mesh.hpp"

namespace Ilum
{
Mesh::Mesh(RHIDevice *device, const std::string &filename):
    p_device(device)
{

}

const std::string &Mesh::GetName() const
{
	return m_name;
}

}        // namespace Ilum