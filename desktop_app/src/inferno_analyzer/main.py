"""Entry point for Inferno Analyzer desktop app."""

import os

os.environ["LIBXRK_BACKEND"] = "rust"

from inferno_analyzer.app import InfernoAnalyzerApp


def main() -> None:
    """Launch the Inferno Analyzer application."""
    app = InfernoAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
