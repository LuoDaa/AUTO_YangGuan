"""Compatibility entrypoint for the localization-first raw-case adapter.

The retired prototype used ``parts[1] + PCA axis + circular fake Margin``.
All public callers now use case_adapter_v2 with the authoritative V3 localization
runtime (support ranking -> rough axis -> Margin -> insertion solve -> frame).
"""

from . import case_adapter_v2 as _impl
from .localization_runtime import localize_case_geometry as _authoritative_localize_case_geometry

# case_adapter_v2 functions resolve this global at call time; install the robust
# runtime before re-exporting its public API. This keeps old imports stable while
# removing the proxy localization path from production.
_impl.localize_case_geometry = _authoritative_localize_case_geometry

from .case_adapter_v2 import *  # noqa: F401,F403,E402
