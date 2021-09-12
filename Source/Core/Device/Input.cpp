#include "Input.hpp"
#include "Window.hpp"

#include "Core/Engine/Context.hpp"

#include <SDL.h>
#include <SDL_syswm.h>

namespace Ilum
{
Input::Input(Context *context):
    TSubsystem<Input>(context)
{
	// Initialize events subsystem
	if (SDL_WasInit(SDL_INIT_EVENTS) != 1)
	{
		if (SDL_InitSubSystem(SDL_INIT_EVENTS) != 0)
		{
			LOG_ERROR("Failed to initialize SDL events subsystem: {}", SDL_GetError());
			return;
		}
	}

	// Initialize controller subsystem
	if (SDL_WasInit(SDL_INIT_GAMECONTROLLER) != 1)
	{
		if (SDL_InitSubSystem(SDL_INIT_GAMECONTROLLER) != 0)
		{
			LOG_ERROR("Failed to initialize SDL controller subsystem: {}", SDL_GetError());
			return;
		}
	}

	m_keys.fill(false);
	m_last_keys.fill(false);

	if (!m_context->hasSubsystem<Window>())
	{
		m_context->addSubsystem<Window>();
	}

	m_context->getSubsystem<Window>()->Event_SDL += std::bind(&Input::onEvent, this, std::placeholders::_1);
}

void Input::onTick(float delta_time)
{
	m_last_keys = m_keys;

	pollMouse();
	pollKeyboard();
}

void Input::onPostTick()
{
	m_mouse_wheel_delta = {0, 0};
}

void Input::pollMouse()
{
	// Get state
	int  x, y;
	auto key_states = SDL_GetGlobalMouseState(&x, &y);

	// Get delta
	m_mouse_delta = {static_cast<int32_t>(x) - m_mouse_position.first,
	                 static_cast<int32_t>(y) - m_mouse_position.second};

	// Get position
	m_mouse_position = {static_cast<int32_t>(x),
	                    static_cast<int32_t>(y)};

	// Get Keys
	m_keys[m_mouse_start_index]                          = (key_states & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
	m_keys[static_cast<size_t>(m_mouse_start_index) + 1] = (key_states & SDL_BUTTON(SDL_BUTTON_MIDDLE)) != 0;
	m_keys[static_cast<size_t>(m_mouse_start_index) + 2] = (key_states & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0;
}

void Input::pollKeyboard()
{
	// Get keyboard state
	const auto *key_states = SDL_GetKeyboardState(nullptr);

	// Function
	m_keys[0]  = key_states[SDL_SCANCODE_F1];
	m_keys[1]  = key_states[SDL_SCANCODE_F2];
	m_keys[2]  = key_states[SDL_SCANCODE_F3];
	m_keys[3]  = key_states[SDL_SCANCODE_F4];
	m_keys[4]  = key_states[SDL_SCANCODE_F5];
	m_keys[5]  = key_states[SDL_SCANCODE_F6];
	m_keys[6]  = key_states[SDL_SCANCODE_F7];
	m_keys[7]  = key_states[SDL_SCANCODE_F8];
	m_keys[8]  = key_states[SDL_SCANCODE_F9];
	m_keys[9]  = key_states[SDL_SCANCODE_F10];
	m_keys[10] = key_states[SDL_SCANCODE_F11];
	m_keys[11] = key_states[SDL_SCANCODE_F12];
	m_keys[12] = key_states[SDL_SCANCODE_F13];
	m_keys[13] = key_states[SDL_SCANCODE_F14];
	m_keys[14] = key_states[SDL_SCANCODE_F15];

	// Numbers
	m_keys[15] = key_states[SDL_SCANCODE_0];
	m_keys[16] = key_states[SDL_SCANCODE_1];
	m_keys[17] = key_states[SDL_SCANCODE_2];
	m_keys[18] = key_states[SDL_SCANCODE_3];
	m_keys[19] = key_states[SDL_SCANCODE_4];
	m_keys[20] = key_states[SDL_SCANCODE_5];
	m_keys[21] = key_states[SDL_SCANCODE_6];
	m_keys[22] = key_states[SDL_SCANCODE_7];
	m_keys[23] = key_states[SDL_SCANCODE_8];
	m_keys[24] = key_states[SDL_SCANCODE_9];

	// Key pad
	m_keys[25] = key_states[SDL_SCANCODE_KP_0];
	m_keys[26] = key_states[SDL_SCANCODE_KP_1];
	m_keys[27] = key_states[SDL_SCANCODE_KP_2];
	m_keys[28] = key_states[SDL_SCANCODE_KP_3];
	m_keys[29] = key_states[SDL_SCANCODE_KP_4];
	m_keys[30] = key_states[SDL_SCANCODE_KP_5];
	m_keys[31] = key_states[SDL_SCANCODE_KP_6];
	m_keys[32] = key_states[SDL_SCANCODE_KP_7];
	m_keys[33] = key_states[SDL_SCANCODE_KP_8];
	m_keys[34] = key_states[SDL_SCANCODE_KP_9];

	// Letters
	m_keys[35] = key_states[SDL_SCANCODE_Q];
	m_keys[36] = key_states[SDL_SCANCODE_W];
	m_keys[37] = key_states[SDL_SCANCODE_E];
	m_keys[38] = key_states[SDL_SCANCODE_R];
	m_keys[39] = key_states[SDL_SCANCODE_T];
	m_keys[40] = key_states[SDL_SCANCODE_Y];
	m_keys[41] = key_states[SDL_SCANCODE_U];
	m_keys[42] = key_states[SDL_SCANCODE_I];
	m_keys[43] = key_states[SDL_SCANCODE_O];
	m_keys[44] = key_states[SDL_SCANCODE_P];
	m_keys[45] = key_states[SDL_SCANCODE_A];
	m_keys[46] = key_states[SDL_SCANCODE_S];
	m_keys[47] = key_states[SDL_SCANCODE_D];
	m_keys[48] = key_states[SDL_SCANCODE_F];
	m_keys[49] = key_states[SDL_SCANCODE_G];
	m_keys[50] = key_states[SDL_SCANCODE_H];
	m_keys[51] = key_states[SDL_SCANCODE_J];
	m_keys[52] = key_states[SDL_SCANCODE_K];
	m_keys[53] = key_states[SDL_SCANCODE_L];
	m_keys[54] = key_states[SDL_SCANCODE_Z];
	m_keys[55] = key_states[SDL_SCANCODE_X];
	m_keys[56] = key_states[SDL_SCANCODE_C];
	m_keys[57] = key_states[SDL_SCANCODE_V];
	m_keys[58] = key_states[SDL_SCANCODE_B];
	m_keys[59] = key_states[SDL_SCANCODE_N];
	m_keys[60] = key_states[SDL_SCANCODE_M];

	// Controls
	m_keys[61] = key_states[SDL_SCANCODE_ESCAPE];
	m_keys[62] = key_states[SDL_SCANCODE_TAB];
	m_keys[63] = key_states[SDL_SCANCODE_LSHIFT];
	m_keys[64] = key_states[SDL_SCANCODE_RSHIFT];
	m_keys[65] = key_states[SDL_SCANCODE_LCTRL];
	m_keys[66] = key_states[SDL_SCANCODE_RCTRL];
	m_keys[67] = key_states[SDL_SCANCODE_LALT];
	m_keys[68] = key_states[SDL_SCANCODE_RALT];
	m_keys[69] = key_states[SDL_SCANCODE_SPACE];
	m_keys[70] = key_states[SDL_SCANCODE_CAPSLOCK];
	m_keys[71] = key_states[SDL_SCANCODE_BACKSPACE];
	m_keys[72] = key_states[SDL_SCANCODE_RETURN];
	m_keys[73] = key_states[SDL_SCANCODE_DELETE];
	m_keys[74] = key_states[SDL_SCANCODE_LEFT];
	m_keys[75] = key_states[SDL_SCANCODE_RIGHT];
	m_keys[76] = key_states[SDL_SCANCODE_UP];
	m_keys[77] = key_states[SDL_SCANCODE_DOWN];
	m_keys[78] = key_states[SDL_SCANCODE_PAGEUP];
	m_keys[79] = key_states[SDL_SCANCODE_PAGEDOWN];
	m_keys[80] = key_states[SDL_SCANCODE_HOME];
	m_keys[81] = key_states[SDL_SCANCODE_END];
	m_keys[82] = key_states[SDL_SCANCODE_INSERT];
}

void Input::onEvent(const SDL_Event &event)
{
	auto event_type = event.type;

	if (event_type == SDL_MOUSEWHEEL)
	{
		onEventMouse(event);
	}

	if (event_type == SDL_CONTROLLERAXISMOTION ||
	    event_type == SDL_CONTROLLERBUTTONDOWN ||
	    event_type == SDL_CONTROLLERBUTTONUP ||
	    event_type == SDL_CONTROLLERDEVICEADDED ||
	    event_type == SDL_CONTROLLERDEVICEREMOVED ||
	    event_type == SDL_CONTROLLERDEVICEREMAPPED ||
	    event_type == SDL_CONTROLLERTOUCHPADDOWN ||
	    event_type == SDL_CONTROLLERTOUCHPADMOTION ||
	    event_type == SDL_CONTROLLERTOUCHPADUP ||
	    event_type == SDL_CONTROLLERSENSORUPDATE)
	{
		onEventController(event);
	}
}

void Input::onEventMouse(const SDL_Event &event)
{
	auto event_type = event.type;

	// Wheel
	if (event_type == SDL_MOUSEWHEEL)
	{
		if (event.wheel.x > 0)
			m_mouse_wheel_delta.first += 1;
		else if (event.wheel.x < 0)
			m_mouse_wheel_delta.first -= 1;
		if (event.wheel.y > 0)
			m_mouse_wheel_delta.second += 1;
		else if (event.wheel.y < 0)
			m_mouse_wheel_delta.second -= 1;
	}
}

void Input::onEventController(const SDL_Event &event)
{
	auto event_type = event.type;

	// Detect controller
	if (!m_controller_connection)
	{
		for (uint32_t i = 0; i < static_cast<uint32_t>(SDL_NumJoysticks()); i++)
		{
			SDL_GameController *controller = SDL_GameControllerOpen(i);
			if (SDL_GameControllerGetAttached(controller) == 1)
			{
				m_controller            = controller;
				m_controller_connection = true;
				LOG_INFO("Detected controller: {}", i);
			}
			else
			{
				LOG_ERROR("Failed to detect controller: {}", SDL_GetError());
			}
		}
		SDL_GameControllerEventState(SDL_ENABLE);
	}

	// Connected
	if (event_type == SDL_CONTROLLERDEVICEADDED)
	{
		// Get first available controller
		for (uint32_t i = 0; i < static_cast<uint32_t>(SDL_NumJoysticks()); i++)
		{
			if (SDL_IsGameController(i))
			{
				SDL_GameController *controller = SDL_GameControllerOpen(i);
				if (SDL_GameControllerGetAttached(controller) == SDL_TRUE)
				{
					m_controller            = controller;
					m_controller_connection = true;
					break;
				}
			}
		}

		if (m_controller_connection)
		{
			LOG_INFO("Controller connected.");
		}
		else
		{
			LOG_ERROR("Failed to connect controller: {}", SDL_GetError());
		}
	}

	// DIsconnected
	if (event_type == SDL_CONTROLLERDEVICEREMOVED)
	{
		m_controller            = nullptr;
		m_controller_connection = false;
		LOG_INFO("Controller disconnected.");
	}

	// Keys
	if (event_type == SDL_CONTROLLERBUTTONDOWN)
	{
		auto button = event.cbutton.button;

		m_keys[m_controller_start_index]                           = button == SDL_CONTROLLER_BUTTON_DPAD_UP;
		m_keys[static_cast<size_t>(m_controller_start_index) + 1]  = button == SDL_CONTROLLER_BUTTON_DPAD_DOWN;
		m_keys[static_cast<size_t>(m_controller_start_index) + 2]  = button == SDL_CONTROLLER_BUTTON_DPAD_LEFT;
		m_keys[static_cast<size_t>(m_controller_start_index) + 3]  = button == SDL_CONTROLLER_BUTTON_DPAD_RIGHT;
		m_keys[static_cast<size_t>(m_controller_start_index) + 4]  = button == SDL_CONTROLLER_BUTTON_A;
		m_keys[static_cast<size_t>(m_controller_start_index) + 5]  = button == SDL_CONTROLLER_BUTTON_B;
		m_keys[static_cast<size_t>(m_controller_start_index) + 6]  = button == SDL_CONTROLLER_BUTTON_X;
		m_keys[static_cast<size_t>(m_controller_start_index) + 7]  = button == SDL_CONTROLLER_BUTTON_Y;
		m_keys[static_cast<size_t>(m_controller_start_index) + 8]  = button == SDL_CONTROLLER_BUTTON_BACK;
		m_keys[static_cast<size_t>(m_controller_start_index) + 9]  = button == SDL_CONTROLLER_BUTTON_GUIDE;
		m_keys[static_cast<size_t>(m_controller_start_index) + 10] = button == SDL_CONTROLLER_BUTTON_START;
		m_keys[static_cast<size_t>(m_controller_start_index) + 11] = button == SDL_CONTROLLER_BUTTON_LEFTSTICK;
		m_keys[static_cast<size_t>(m_controller_start_index) + 12] = button == SDL_CONTROLLER_BUTTON_RIGHTSTICK;
		m_keys[static_cast<size_t>(m_controller_start_index) + 13] = button == SDL_CONTROLLER_BUTTON_LEFTSHOULDER;
		m_keys[static_cast<size_t>(m_controller_start_index) + 14] = button == SDL_CONTROLLER_BUTTON_RIGHTSHOULDER;
		m_keys[static_cast<size_t>(m_controller_start_index) + 15] = button == SDL_CONTROLLER_BUTTON_MISC1;
		m_keys[static_cast<size_t>(m_controller_start_index) + 16] = button == SDL_CONTROLLER_BUTTON_PADDLE1;
		m_keys[static_cast<size_t>(m_controller_start_index) + 17] = button == SDL_CONTROLLER_BUTTON_PADDLE2;
		m_keys[static_cast<size_t>(m_controller_start_index) + 18] = button == SDL_CONTROLLER_BUTTON_PADDLE3;
		m_keys[static_cast<size_t>(m_controller_start_index) + 19] = button == SDL_CONTROLLER_BUTTON_PADDLE4;
		m_keys[static_cast<size_t>(m_controller_start_index) + 20] = button == SDL_CONTROLLER_BUTTON_TOUCHPAD;
	}
	else
	{
		for (uint32_t i = m_controller_start_index; i < m_controller_start_index + 21; i++)
		{
			m_keys[i] = false;
		}
	}

	// Axes
	if (event_type == SDL_CONTROLLERAXISMOTION)
	{
		SDL_ControllerAxisEvent event_axis = event.caxis;

		switch (event_axis.axis)
		{
			case SDL_CONTROLLER_AXIS_LEFTX:
				m_controller_thumb_left.first = static_cast<float>(event_axis.value) / 32768.f;
				break;
			case SDL_CONTROLLER_AXIS_LEFTY:
				m_controller_thumb_left.second = static_cast<float>(event_axis.value) / 32768.f;
				break;
			case SDL_CONTROLLER_AXIS_RIGHTX:
				m_controller_thumb_right.first = static_cast<float>(event_axis.value) / 32768.f;
				break;
			case SDL_CONTROLLER_AXIS_RIGHTY:
				m_controller_thumb_right.second = static_cast<float>(event_axis.value) / 32768.f;
				break;
			case SDL_CONTROLLER_AXIS_TRIGGERLEFT:
				m_controller_trigger_left = static_cast<float>(event_axis.value) / 32768.f;
				break;
			case SDL_CONTROLLER_AXIS_TRIGGERRIGHT:
				m_controller_trigger_right = static_cast<float>(event_axis.value) / 32768.f;
				break;
			default:
				break;
		}
	}
}

bool Input::getKey(const KeyCode key) const
{
	return m_keys[static_cast<uint32_t>(key)];
}

bool Input::getKeyDown(const KeyCode key) const
{
	return getKey(key) && !m_last_keys[static_cast<uint32_t>(key)];
}

bool Input::getKeyUp(const KeyCode key) const
{
	return !getKey(key) && m_last_keys[static_cast<uint32_t>(key)];
}

void Input::setMouseCursorVisible(const bool visible)
{
	if (visible == m_mouse_cursor_visible)
	{
		return;
	}

	if (visible)
	{
		if (SDL_ShowCursor(SDL_ENABLE) != 0)
		{
			LOG_ERROR("Failed to show cursor");
			return;
		}
	}
	else
	{
		if (SDL_ShowCursor(SDL_DISABLE) != 1)
		{
			LOG_ERROR("Failed to hide cursor");
			return;
		}
	}

	m_mouse_cursor_visible = visible;
}

bool Input::getMouseCursorVisible() const
{
	return m_mouse_cursor_visible;
}

void Input::setMousePosition(uint32_t x, uint32_t y)
{
	if (SDL_WarpMouseGlobal(static_cast<int>(x), static_cast<int>(y)) != 0)
	{
		LOG_ERROR("Failed to set mouse position");
		return;
	}

	m_mouse_position = {x, y};
}
const std::pair<int32_t, int32_t> &Input::getMousePosition() const
{
	return m_mouse_position;
}

const std::pair<int32_t, int32_t> &Input::getMouseDelta() const
{
	return m_mouse_delta;
}

const std::pair<int32_t, int32_t> &Input::getMouseWheelDelta() const
{
	return m_mouse_wheel_delta;
}

bool Input::controllerConnection()
{
	return m_controller_connection;
}

const std::pair<float, float> &Input::getControllerThumbStickLeft() const
{
	return m_controller_thumb_left;
}

const std::pair<float, float> &Input::getControllerThumbStickRight() const
{
	return m_controller_thumb_right;
}

float Input::getControllerTriggerLeft() const
{
	return m_controller_trigger_left;
}

float Input::getControllerTriggerRight() const
{
	return m_controller_trigger_right;
}

bool Input::controllerVibrate(const float left_motor_speed, const float right_motor_speed) const
{
	if (!m_controller_connection)
	{
		return false;
	}

	auto left_speed               = left_motor_speed < 0.f ? 0.f : left_motor_speed;
	left_speed                    = left_speed > 1.f ? 1.f : left_speed;
	uint16_t low_frequency_rumble = static_cast<uint16_t>(left_speed * 65535);

	auto right_speed               = right_motor_speed < 0.f ? 0.f : right_motor_speed;
	right_speed                    = right_speed > 1.f ? 1.f : right_speed;
	uint16_t high_frequency_rumble = static_cast<uint16_t>(right_speed * 65535);

	uint32_t duration_ms = 0xFFFFFFFF;

	if (SDL_GameControllerRumble(m_controller, low_frequency_rumble, high_frequency_rumble, duration_ms) == -1)
	{
		LOG_ERROR("Failed to vibrate controller");
		return false;
	}

	return true;
}
}        // namespace Ilum