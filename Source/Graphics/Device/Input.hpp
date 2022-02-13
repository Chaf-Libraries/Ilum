#pragma once

#include <Core/Event.hpp>

#include <array>

union SDL_Event;
struct _SDL_GameController;
typedef _SDL_GameController SDL_GameController;

namespace Ilum::Graphics
{
enum class KeyCode
{
	// Keyboard
	F1,
	F2,
	F3,
	F4,
	F5,
	F6,
	F7,
	F8,
	F9,
	F10,
	F11,
	F12,
	F13,
	F14,
	F15,
	Alpha0,
	Alpha1,
	Alpha2,
	Alpha3,
	Alpha4,
	Alpha5,
	Alpha6,
	Alpha7,
	Alpha8,
	Alpha9,
	Keypad0,
	Keypad1,
	Keypad2,
	Keypad3,
	Keypad4,
	Keypad5,
	Keypad6,
	Keypad7,
	Keypad8,
	Keypad9,
	Q,
	W,
	E,
	R,
	T,
	Y,
	U,
	I,
	O,
	P,
	A,
	S,
	D,
	F,
	G,
	H,
	J,
	K,
	L,
	Z,
	X,
	C,
	V,
	B,
	N,
	M,
	Esc,
	Tab,
	Shift_Left,
	Shift_Right,
	Ctrl_Left,
	Ctrl_Right,
	Alt_Left,
	Alt_Right,
	Space,
	CapsLock,
	Backspace,
	Enter,
	Delete,
	Arrow_Left,
	Arrow_Right,
	Arrow_Up,
	Arrow_Down,
	Page_Up,
	Page_Down,
	Home,
	End,
	Insert,

	// Mouse
	Click_Left,
	Click_Middle,
	Click_Right,

	// Controller
	DPad_Up,
	DPad_Down,
	DPad_Left,
	DPad_Right,
	Button_A,
	Button_B,
	Button_X,
	Button_Y,
	Back,
	Guide,
	Start,
	Left_Stick,
	Right_Stick,
	Left_Shoulder,
	Right_Shoulder,
	Misc1,           // Xbox Series X share button, PS5 microphone button, Nintendo Switch Pro capture button
	Paddle1,         // Xbox Elite paddle P1
	Paddle2,         // Xbox Elite paddle P3
	Paddle3,         // Xbox Elite paddle P2
	Paddle4,         // Xbox Elite paddle P4
	Touchpad,        // PS4/PS5 touchpad button
};

class Input
{
  public:
	Input();
	~Input() = default;

	static Input &GetInstance();

	void OnUpdate();

	// Polling driven input
	void PollMouse();
	void PollKeyboard();

	// Event driven input
	void OnEvent(const SDL_Event &event);
	void OnEventMouse(const SDL_Event &event);
	void OnEventController(const SDL_Event &event);

	// Keys
	bool GetKey(const KeyCode key) const;
	bool GetKeyDown(const KeyCode key) const;
	bool GetKeyUp(const KeyCode key) const;

	// Mouse
	void                               SetMouseCursorVisible(const bool visible);
	bool                               GetMouseCursorVisible() const;
	void                               SetMousePosition(uint32_t x, uint32_t y);
	const std::pair<int32_t, int32_t> &GetMousePosition() const;
	const std::pair<int32_t, int32_t> &GetMouseDelta() const;
	const std::pair<int32_t, int32_t> &GetMouseWheelDelta() const;

	// Controller
	bool                           ControllerConnection();
	const std::pair<float, float> &GetControllerThumbStickLeft() const;
	const std::pair<float, float> &GetControllerThumbStickRight() const;
	float                          GetControllerTriggerLeft() const;
	float                          GetControllerTriggerRight() const;
	bool                           ControllerVibrate(const float left_motor_speed, const float right_motor_speed) const;

  private:
	// Keys
	std::array<bool, 107> m_keys                   = {false};
	std::array<bool, 107> m_last_keys              = {false};
	uint32_t              m_mouse_start_index      = 83;
	uint32_t              m_controller_start_index = 86;

	// Mouse
	std::pair<int32_t, int32_t> m_mouse_position       = {0, 0};
	std::pair<int32_t, int32_t> m_mouse_delta          = {0, 0};
	std::pair<int32_t, int32_t> m_mouse_wheel_delta    = {0, 0};
	bool                        m_mouse_cursor_visible = true;

	// Controller
	SDL_GameController *    m_controller               = nullptr;
	bool                    m_controller_connection    = false;
	std::pair<float, float> m_controller_thumb_left    = {0.f, 0.f};
	std::pair<float, float> m_controller_thumb_right   = {0.f, 0.f};
	float                   m_controller_trigger_left  = 0.f;
	float                   m_controller_trigger_right = 0.f;
};
}        // namespace Ilum