"""Entry point for Driver Consistency Analyzer desktop app."""

import os

os.environ["LIBXRK_BACKEND"] = "rust"

from driver_consistency_analyzer.app import DriverConsistencyApp


def main() -> None:
    """Launch the Driver Consistency Analyzer application."""
    app = DriverConsistencyApp()
    app.mainloop()


if __name__ == "__main__":
    main()
