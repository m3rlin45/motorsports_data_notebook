"""Entry point for Suspension Analyzer desktop app."""

import os

os.environ["LIBXRK_BACKEND"] = "rust"

from suspension_analyzer.app import SuspensionAnalyzerApp


def main() -> None:
    """Launch the Suspension Analyzer application."""
    app = SuspensionAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
