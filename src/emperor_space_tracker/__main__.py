"""Enable ``python -m emperor_space_tracker``.

The systemd unit's ``ExecStart`` uses this, and it is also the supported way to
run the daemon under any other supervisor (runit, s6, a container entrypoint).
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
