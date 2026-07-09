"""Exceptions for the data layer.

The project's core data-integrity rule is: never fabricate or silently patch
over missing or broken data. These exceptions are how that rule is enforced
in code -- any ambiguity about data quality must surface as a loud failure,
not a silently interpolated or zero-filled value.
"""

from __future__ import annotations


class DataFetchError(RuntimeError):
    """A vendor API call failed, or returned data that cannot be trusted."""


class ConfigError(RuntimeError):
    """Configuration (YAML) is missing, malformed, or internally inconsistent."""


class DataGapError(DataFetchError):
    """A series has a larger-than-expected gap between observations.

    A single missing day is normal (holiday). A multi-week gap usually means
    the vendor silently stopped publishing, or a request parameter is wrong --
    both are failures worth stopping the pipeline for, not skipping past.
    """
