"""Configuration management for claude-stt."""

import logging
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import tomllib as tomli
except ImportError:
    try:
        import tomli
    except ImportError:
        tomli = None

try:
    import tomli_w
except ImportError:
    tomli_w = None

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """claude-stt configuration."""

    # Hotkey settings
    hotkey: str = "ctrl+shift+space"
    improve_hotkey: str = "cmd+alt+d"
    improve_model: str = "haiku"
    mode: Literal["push-to-talk", "toggle"] = "toggle"

    # Engine settings
    stt_model: str = "CohereLabs/cohere-transcribe-03-2026"

    # Audio settings
    sample_rate: int = 16000
    max_recording_seconds: int = 300  # 5 minutes
    audio_device: str | int | None = None  # None = system default

    # Output settings
    output_mode: Literal["injection", "clipboard", "auto"] = "auto"

    # Feedback settings
    sound_effects: bool = True

    # Use Shift+Enter for newlines (soft newline) instead of Enter
    # Only the final trailing newline (if any) becomes a real Enter
    soft_newlines: bool = True

    # Language for transcription ("auto" = detect, or ISO 639-1 code e.g. "en")
    language: str = "auto"

    # Apps to exclude from hotkey capture (case-insensitive substring match)
    excluded_apps: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.excluded_apps is None:
            self.excluded_apps = []

    @classmethod
    def get_config_dir(cls) -> Path:
        """Get the configuration directory path."""
        override = os.environ.get("CLAUDE_STT_CONFIG_DIR")
        if override:
            return Path(override).expanduser()
        return Path.home() / ".config" / "claude-stt"

    @classmethod
    def get_config_path(cls) -> Path:
        """Get the configuration file path."""
        return cls.get_config_dir() / "config.toml"

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from file, or return defaults."""
        config_path = cls.get_config_path()
        if not config_path.exists():
            return cls()

        if tomli is None:
            logger.warning("tomli not installed; using default config")
            return cls().validate()

        try:
            with open(config_path, "rb") as f:
                data = tomli.load(f)

            stt_config = data.get("claude-stt", {})
            config = cls(
                hotkey=stt_config.get("hotkey", cls.hotkey),
                mode=stt_config.get("mode", cls.mode),
                stt_model=stt_config.get("stt_model", cls.stt_model),
                sample_rate=stt_config.get("sample_rate", cls.sample_rate),
                max_recording_seconds=stt_config.get(
                    "max_recording_seconds", cls.max_recording_seconds
                ),
                audio_device=stt_config.get("audio_device", cls.audio_device),
                output_mode=stt_config.get("output_mode", cls.output_mode),
                sound_effects=stt_config.get("sound_effects", cls.sound_effects),
                improve_hotkey=stt_config.get("improve_hotkey", cls.improve_hotkey),
                improve_model=stt_config.get("improve_model", cls.improve_model),
                soft_newlines=stt_config.get("soft_newlines", cls.soft_newlines),
                language=stt_config.get("language", cls.language),
                excluded_apps=stt_config.get("excluded_apps", []),
            )
            return config.validate()
        except Exception:
            logger.exception("Failed to load config; using defaults")
            return cls().validate()

    def save(self) -> bool:
        """Save configuration to file."""
        if tomli_w is None:
            logger.warning("tomli-w not installed; config not saved")
            return False

        config_path = self.get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "claude-stt": {
                "hotkey": self.hotkey,
                "mode": self.mode,
                "stt_model": self.stt_model,
                "sample_rate": self.sample_rate,
                "max_recording_seconds": self.max_recording_seconds,
                "audio_device": self.audio_device,
                "output_mode": self.output_mode,
                "sound_effects": self.sound_effects,
                "improve_hotkey": self.improve_hotkey,
                "improve_model": self.improve_model,
                "soft_newlines": self.soft_newlines,
                "language": self.language,
                "excluded_apps": self.excluded_apps,
            }
        }

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                delete=False,
                dir=str(config_path.parent),
            ) as handle:
                temp_file = Path(handle.name)
                tomli_w.dump(data, handle)
            os.replace(temp_file, config_path)
            return True
        except Exception:
            logger.exception("Failed to save config")
            return False
        finally:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def validate(self) -> "Config":
        """Validate and normalize configuration values."""
        if not isinstance(self.hotkey, str) or not self.hotkey.strip():
            logger.warning("Invalid hotkey; defaulting to 'ctrl+shift+space'")
            self.hotkey = "ctrl+shift+space"

        if self.mode not in ("push-to-talk", "toggle"):
            logger.warning("Invalid mode '%s'; defaulting to 'toggle'", self.mode)
            self.mode = "toggle"

        if not isinstance(self.stt_model, str) or not self.stt_model.strip():
            logger.warning("Invalid stt_model; defaulting to Cohere Transcribe")
            self.stt_model = "CohereLabs/cohere-transcribe-03-2026"

        if self.output_mode not in ("injection", "clipboard", "auto"):
            logger.warning("Invalid output_mode '%s'; defaulting to 'auto'", self.output_mode)
            self.output_mode = "auto"

        if not isinstance(self.sound_effects, bool):
            if isinstance(self.sound_effects, str):
                self.sound_effects = self.sound_effects.strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
            else:
                self.sound_effects = bool(self.sound_effects)

        try:
            self.max_recording_seconds = int(self.max_recording_seconds)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid max_recording_seconds '%s'; defaulting to %s",
                self.max_recording_seconds,
                Config.max_recording_seconds,
            )
            self.max_recording_seconds = Config.max_recording_seconds

        if self.max_recording_seconds < 1:
            logger.warning("max_recording_seconds too low; clamping to 1")
            self.max_recording_seconds = 1
        elif self.max_recording_seconds > 600:
            logger.warning("max_recording_seconds too high; clamping to 600")
            self.max_recording_seconds = 600

        if self.sample_rate != 16000:
            logger.warning("sample_rate %s not supported; forcing 16000", self.sample_rate)
            self.sample_rate = 16000

        return self


def get_platform() -> str:
    """Get the current platform identifier."""
    return {
        "Darwin": "macos",
        "Linux": "linux",
        "Windows": "windows",
    }.get(platform.system(), "unknown")


def is_wayland() -> bool:
    """Check if running under Wayland on Linux."""
    if get_platform() != "linux":
        return False
    return os.environ.get("XDG_SESSION_TYPE") == "wayland"
