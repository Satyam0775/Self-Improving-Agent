"""Small formatting helpers shared by the agent, tools and evaluator."""

from __future__ import annotations

import datetime as dt
import re


def format_date(d: dt.date) -> str:
    """'Wednesday 4 March 2026' (portable: avoids %-d, which fails on Windows)."""
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B %Y')}"


def format_time(hhmm: str) -> str:
    """'16:00' -> '4:00 PM'."""
    hour, minute = (int(p) for p in hhmm.split(":"))
    suffix = "AM" if hour < 12 else "PM"
    hour12 = hour % 12 or 12
    return f"{hour12}:{minute:02d} {suffix}"


def slot_label(date_iso: str, hhmm: str) -> str:
    return f"{format_date(dt.date.fromisoformat(date_iso))} at {format_time(hhmm)}"


def join_natural(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTH_NAMES = "|".join(m.title() for m in _MONTHS)
_DAY_NAMES = "|".join(_WEEKDAYS)
_WEEKDAY_DATE_RE = re.compile(
    rf"\b(?P<wd>{_DAY_NAMES}),?\s+(?:(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m1>{_MONTH_NAMES})"
    rf"|(?P<m2>{_MONTH_NAMES})\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?)(?:,?\s+(?P<y>\d{{4}}))?\b"
)


def correct_weekdays(text: str, today: dt.date) -> tuple[str, list[str]]:
    """Fix any 'Weekday D Month [Year]' mention whose weekday does not match the real calendar.

    The model is told to copy weekday/date pairs from tool results, but it can still miscalculate
    (e.g. 'Monday 18 October' when 18 October is a Sunday). Returns (text, list of corrections).
    """
    fixes: list[str] = []

    def fix(m: re.Match[str]) -> str:
        month = _MONTHS[(m["m1"] or m["m2"]).lower()]
        day = int(m["d1"] or m["d2"])
        years = [int(m["y"])] if m["y"] else [today.year, today.year + 1]
        for year in years:
            try:
                real = dt.date(year, month, day)
            except ValueError:
                continue
            if m["y"] or real >= today:
                actual = _WEEKDAYS[real.weekday()]
                if actual == m["wd"]:
                    return m[0]
                fixes.append(f"{m[0]} -> {actual}")
                return actual + m[0][len(m["wd"]):]
        return m[0]

    return _WEEKDAY_DATE_RE.sub(fix, text), fixes
