#pragma once

#include "Core/Engine/Eventing/Event.hpp"
#include "Core/Engine/PCH.hpp"
#include "Core/Engine/Subsystem.hpp"

union SDL_Event;
struct _SDL_GameController;
typedef _SDL_GameController SDL_GameController;

namespace Ilum
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

class Input : public TSubsystem<Input>
{
  public:
	Input(Context *context = nullptr);
	~Input() = default;

	virtual void onTick(float delta_time) override;
	virtual void onPostTick() override;

	// Polling driven input
	void pollMouse();
	void pollKeyboard();

	// Event driven input
	void onEvent(const SDL_Event &event);
	void onEventMouse(const SDL_Event &event);
	void onEventController(const SDL_Event &event);

	// Keys
	bool getKey(const KeyCode key) const;
	bool getKeyDown(const KeyCode key) const;
	bool getKeyUp(const KeyCode key) const;

	// Mouse
	void                               setMouseCursorVisible(const bool visible);
	bool                               getMouseCursorVisible() const;
	void                               setMousePosition(uint32_t x, uint32_t y);
	const std::pair<int32_t, int32_t> &getMousePosition() const;
	const std::pair<int32_t, int32_t> &getMouseDelta() const;
	const std::pair<int32_t, int32_t> &getMouseWheelDelta() const;

	// Controller
	bool                           controllerConnection();
	const std::pair<float, float> &getControllerThumbStickLeft() const;
	const std::pair<float, float> &getControllerThumbStickRight() const;
	float                          getControllerTriggerLeft() const;
	float                          getControllerTriggerRight() const;
	bool                           controllerVibrate(const float left_motor_speed, const float right_motor_speed) const;

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