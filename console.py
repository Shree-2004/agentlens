"""Make stdout safe for whatever the judge hands back.

Windows consoles default to cp1252, and models routinely emit characters
outside it -- a narrow no-break space (U+202F) inside "step 2" is what
first crashed analyze.py here, *after* the diagnosis had already been
written to disk. The result was a traceback on a run that had completely
succeeded, which is a bad way to learn that your API call worked.

Printing is never the point of these scripts, so it should never be the
thing that fails them.
"""

from __future__ import annotations

import sys


def enable_utf8_output() -> None:
    """Switch stdout/stderr to UTF-8, replacing anything unencodable.

    Call once at the top of a CLI's main(). A no-op where the streams are
    already UTF-8 (most non-Windows terminals) or don't support
    reconfigure (a pipe wrapped by something unusual).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Detached or already-wrapped stream: leave it alone rather than
            # taking down a run that otherwise worked.
            pass
