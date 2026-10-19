"""Stable public entrypoint for the CrownFlow-derived raw-case adapter.

All production dependencies are wired explicitly inside case_adapter_v2.  This
module only preserves the external import path; it does not monkey-patch
functions at import time.
"""

from . import case_adapter_v2 as _impl
from .case_adapter_v2 import *  # noqa: F401,F403,E402
