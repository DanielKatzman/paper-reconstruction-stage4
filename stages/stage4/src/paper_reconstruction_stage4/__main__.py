from .process_diagnostics import configure

configure()

from .cli import main  # Native fault handling must precede numerical imports.

raise SystemExit(main())
