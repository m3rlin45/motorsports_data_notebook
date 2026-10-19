#!/usr/bin/env bash
# Open a URL in the Windows default browser from WSL.
#
# Used as $BROWSER for `just run`. Python's webbrowser module would otherwise
# go through xdg-open, whose text/html handler may be anything (an editor, say).
# wslu 3.2.3 prints spurious "WSL Interoperability is disabled" noise on current
# WSL2 kernels while still working, so its output is dropped and only a real
# failure (non-zero exit) is reported.
if wslview "$@" >/dev/null 2>&1; then
    exit 0
fi
echo "open_browser.sh: wslview failed to open $*" >&2
exit 1
