"""Compatibility import for the corrected localization-first case adapter.

The first raw-case adapter used ``parts[1] + PCA axis + circular fake Margin``.
That implementation is intentionally retired because it can rotate the template
away from the true abutment/insertion direction. All callers now receive V2.
"""

from .case_adapter_v2 import *  # noqa: F401,F403
