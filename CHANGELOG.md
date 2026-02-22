# Changelog

All notable changes to this project will be documented here.

## [0.2.0] - 2026-02-22

### Fixed
- Fixed garbled text output on COSMIC/Wayland. `wtype 0.4` sends mangled
  keycodes via the `zwp_virtual_keyboard_v1` protocol on COSMIC compositor,
  producing random characters (e.g. `1234567345892234O7-9=`) instead of the
  transcribed text.
- Fixed `wl-copy` blocking the daemon. `wl-copy` stays alive as the Wayland
  clipboard owner, so calling it with `subprocess.run()` would hang until
  another clipboard operation replaced it. Now launched via `Popen`
  (non-blocking).

### Changed
- Wayland text injection now uses `wl-copy` + `xdotool key ctrl+v` (clipboard
  paste) instead of direct `wtype` character injection. This works reliably
  across compositors including COSMIC, GNOME, and KDE.
- Falls back to `wtype -M ctrl -k v -m ctrl` if `xdotool` is not available,
  and to raw `wtype` character injection if `wl-copy` is not installed.
- Added diagnostic logging to the output path showing which injection method
  is selected and the Wayland session state.

### Added
- End-to-end Whisper transcription tests (`test_whisper_transcription.py`)
  using a real LibriSpeech audio sample downloaded from HuggingFace CDN.
  Verifies the model returns correct English text, isn't garbled, and handles
  edge cases (silence, very short audio).

## [0.1.0] - 2026-01-14
- Initial open source release.
