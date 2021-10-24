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
	};
}