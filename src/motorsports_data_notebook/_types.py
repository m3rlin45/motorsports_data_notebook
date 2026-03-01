"""Shared type aliases for motorsports_data_notebook."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libxrk.base import LogFile as AimLogFile
    from libibt.base import LogFile as IbtLogFile

    LogFile = AimLogFile | IbtLogFile
