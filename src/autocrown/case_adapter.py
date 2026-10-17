"""Compatibility entrypoint for the localization-first raw-case adapter.

The retired prototype used ``parts[1] + PCA axis + circular fake Margin``.
All public callers now use case_adapter_v2 with the authoritative localization
runtime and library-referenced template normalization.
"""

from . import case_adapter_v2 as _impl
from .localization_runtime import localize_case_geometry as _authoritative_localize_case_geometry
from .template_normalization import canonicalize_full_tooth as _authoritative_canonicalize_full_tooth

# case_adapter_v2 functions resolve these globals at call time. Install the robust
# implementations before re-exporting its public API so old imports remain stable.
_impl.localize_case_geometry = _authoritative_localize_case_geometry
_impl._canonicalize_full_tooth = _authoritative_canonicalize_full_tooth

from .case_adapter_v2 import *  # noqa: F401,F403,E402
