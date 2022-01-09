#pragma once

#include "Utils/PCH.hpp"

#include "Scene/System.hpp"

namespace Ilum::sym
{
	class TransformUpdate :public System
	{
	  public:
	    TransformUpdate() = default;

		~TransformUpdate() = default;

		virtual void run() override;

	  private:
	    uint32_t m_motionless_count = 0;
	};
}