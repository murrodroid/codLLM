"""Pydantic schemas for structured ICD10h classification output."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

_ICD10H_RE = re.compile(r"^[A-Z]\d{2}\.\d{3}$")


class ICD10hCode(BaseModel):
    """A single validated ICD10h code (e.g. ``A00.000``)."""

    code: str

    @field_validator("code")
    @classmethod
    def _check_format(cls, v: str) -> str:
        v = v.strip()
        if not _ICD10H_RE.match(v):
            raise ValueError(
                f"'{v}' does not match ICD10h format [A-Z]\\d{{2}}.\\d{{3}}"
            )
        return v


class ICD10hCodeList(BaseModel):
    """An ordered list of one or more ICD10h codes."""

    codes: list[ICD10hCode] = Field(min_length=1)

    @classmethod
    def from_raw_string(cls, raw: str, separator: str = " | ") -> ICD10hCodeList:
        """Parse a raw model-output string into a validated code list.

        Raises ``ValidationError`` if *any* token is invalid.
        """
        tokens = [t.strip() for t in raw.split(separator) if t.strip()]
        return cls(codes=[ICD10hCode(code=t) for t in tokens])

    def to_label_string(self, separator: str = " | ") -> str:
        """Serialize back to the same delimited format."""
        return separator.join(c.code for c in self.codes)
