"""Deterministic text-understanding helpers.

Used for (a) the safety gates that must never depend on an LLM (emergency
detection, explicit-confirmation detection) and (b) the rule-based simulated
model used by the offline evaluation.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Optional

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))

_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DAY_MONTH_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\b")
_MONTH_DAY_RE = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b")
_WEEKDAY_RE = re.compile(r"\b(" + "|".join(WEEKDAYS) + r")\b")
_IN_DAYS_RE = re.compile(r"\bin (\d{1,2}) days?\b")
_AMBIGUOUS_RE = re.compile(
    r"\b(soon|sometime|some time|whenever|asap|as soon as possible|next week|this week|"
    r"later|this month|next month|flexible|any day|some day|someday)\b"
)
_ID_RE = re.compile(r"\b(apt|pat|slot)-?[\w-]*", re.IGNORECASE)


@dataclass
class DateParse:
    date: Optional[str] = None  # ISO date
    ambiguous: bool = False


def parse_date(text: str, today: dt.date) -> DateParse:
    t = text.lower()
    m = _ISO_RE.search(t)
    if m:
        try:
            return DateParse(dt.date(int(m[1]), int(m[2]), int(m[3])).isoformat())
        except ValueError:
            return DateParse(None, True)
    if "day after tomorrow" in t:
        return DateParse((today + dt.timedelta(days=2)).isoformat())
    if re.search(r"\btomorrow\b", t):
        return DateParse((today + dt.timedelta(days=1)).isoformat())
    if re.search(r"\b(today|tonight)\b", t):
        return DateParse(today.isoformat())
    m = _IN_DAYS_RE.search(t)
    if m:
        return DateParse((today + dt.timedelta(days=int(m[1]))).isoformat())
    m = _DAY_MONTH_RE.search(t)
    if m:
        day, month = int(m[1]), MONTHS[m[2]]
    else:
        m2 = _MONTH_DAY_RE.search(t)
        day, month = (int(m2[2]), MONTHS[m2[1]]) if m2 else (None, None)
    if day is not None:
        for year in (today.year, today.year + 1):
            try:
                candidate = dt.date(year, month, day)
            except ValueError:
                return DateParse(None, True)
            if candidate >= today:
                return DateParse(candidate.isoformat())
        return DateParse(None, True)
    m = _WEEKDAY_RE.search(t)
    if m:
        delta = (WEEKDAYS[m[1]] - today.weekday()) % 7 or 7  # next occurrence, never today
        return DateParse((today + dt.timedelta(days=delta)).isoformat())
    if _AMBIGUOUS_RE.search(t):
        return DateParse(None, True)
    return DateParse()


def parse_time_preference(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r"\bmorning\b|\bmornings\b", t):
        return "morning"
    if re.search(r"\bafternoon\b|\bafternoons\b", t):
        return "afternoon"
    if re.search(r"\b(any ?time|either|no preference|doesn'?t matter|don'?t mind)\b", t):
        return "any"
    return None


def _strip_dates_and_ids(t: str) -> str:
    t = _ISO_RE.sub(" ", t)
    t = _DAY_MONTH_RE.sub(" ", t)
    t = _MONTH_DAY_RE.sub(" ", t)
    t = _IN_DAYS_RE.sub(" ", t)
    return _ID_RE.sub(" ", t)


def _to_24h(hour: int, minute: int, meridiem: Optional[str]) -> Optional[str]:
    if minute > 59:
        return None
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridiem.startswith("p") else 0)
    else:
        if hour > 23:
            return None
        if 1 <= hour <= 6:  # clinic hours: "4" means 4 PM
            hour += 12
        elif hour == 0:
            return None
    return f"{hour:02d}:{minute:02d}"


def parse_time(text: str) -> Optional[str]:
    """Return 'HH:MM' for an explicit clock time in ``text``, else None."""
    t = _strip_dates_and_ids(text.lower())
    m = re.search(r"\b(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)?(?![a-z])", t)
    if m:
        return _to_24h(int(m[1]), int(m[2]), (m[3] or "").replace(".", "") or None)
    m = re.search(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)(?![a-z])", t)
    if m:
        return _to_24h(int(m[1]), 0, m[2].replace(".", ""))
    m = re.search(r"\b(\d{1,2})\s*o'?clock\b", t)
    if m:
        return _to_24h(int(m[1]), 0, None)
    m = re.search(r"\bat\s+(\d{1,2})\b", t)
    if m:
        return _to_24h(int(m[1]), 0, None)
    m = re.fullmatch(r"\s*(\d{1,2})\s*[.!?]*\s*", t)
    if m:
        return _to_24h(int(m[1]), 0, None)
    return None


def find_appointment_id(text: str) -> Optional[str]:
    m = re.search(r"\bapt-?\s?(\d{1,6})\b", text, re.IGNORECASE)
    if not m:
        return None
    n = int(m[1])
    return f"APT-{n:05d}" if n <= 99999 else None


def mentions_consultation_type(text: str) -> bool:
    return bool(re.search(r"\b(general|consultation|check-?up|routine|regular|standard|gp)\b", text.lower()))


# -- confirmation detection ------------------------------------------------
_NEG_WORDS = re.compile(
    r"\b(no|nope|nah|not|don'?t|dont|do not|never|stop|wait|hold on|instead|actually|but|change|different|rather)\b"
)
_AFFIRM_START = re.compile(
    r"^(yes|yeah|yep|yup|sure|ok|okay|confirm|confirmed|correct|absolutely|definitely|of course|"
    r"please do|go ahead|do it|that works|that's right|sounds good|affirmative)\b"
)
_NEG_START = re.compile(
    r"^(no|nope|nah|never ?mind|not now|not yet|don'?t|do not|stop|cancel that|"
    r"i changed my mind|changed my mind|i'?d rather not)\b"
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text.lower())).strip()


def is_affirmative(text: str) -> bool:
    """Strict: a short message that starts with a clear yes and contains no hedging."""
    t = _norm(text)
    if not t or len(t.split()) > 8:
        return False
    if _NEG_WORDS.search(t):
        return False
    return bool(_AFFIRM_START.match(t))


def is_negative(text: str) -> bool:
    t = _norm(text)
    return bool(_NEG_START.match(t)) or "changed my mind" in t or bool(re.search(r"\b(don'?t|do not) (book|cancel|confirm)\b", t))


def detect_intent(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r"\b(cancel|cancellation|call off)\b", t):
        return "cancel"
    if re.search(r"\b(details|status|look ?up|view|show me my|when is my|what time is my)\b", t):
        return "view"
    if re.search(r"\b(book|schedule|appointment|consultation|doctor|visit|reserve|slot|see someone)\b", t):
        return "book"
    return None


# -- emergency detection ---------------------------------------------------
_EMERGENCY_PATTERNS = [
    r"chest (pain|pressure|tightness)",
    r"(can'?t|cannot|can not|unable to|trouble|difficulty|struggling to|hard to) breath",
    r"short(ness)? of breath",
    r"(not|stopped|stop) breathing",
    r"heart attack",
    r"\bstroke\b",
    r"face (is )?droop",
    r"slurred speech",
    r"sudden (weakness|numbness|confusion|loss of vision|vision loss)",
    r"(severe|heavy|uncontrol\w*) bleeding",
    r"bleeding (heavily|a lot|badly|won'?t stop)",
    r"(coughing|vomiting|throwing up) (up )?blood",
    r"\bunconscious\b|\bpassed out\b|\bfainted\b|\bcollapsed\b",
    r"\bseizure",
    r"overdos",
    r"poisoned|swallowed (bleach|poison)",
    r"anaphyla|throat (is )?(closing|swelling)|severe allergic",
    r"suicid|kill myself|end my life|want to die|hurt myself|harm myself",
    r"severe (chest |abdominal |stomach )?pain",
    r"\bemergency\b|\bambulance\b|\b911\b",
]
_EMERGENCY_RE = re.compile("|".join(f"(?:{p})" for p in _EMERGENCY_PATTERNS))
_NEGATION_BEFORE = re.compile(r"\b(no|not|without|never|denies|isn'?t|nothing like|don'?t have|do not have|haven'?t had)\b")
_NOT_EMERGENCY_RE = re.compile(
    r"(not|isn'?t|no longer|wasn'?t) (an? )?(emergency|urgent)|false alarm|"
    r"i'?m (fine|okay|ok|safe)\b|symptoms (are )?(gone|resolved)|i (was|have been|got) (seen|treated)"
)


def detect_emergency(text: str) -> bool:
    t = text.lower()
    for m in _EMERGENCY_RE.finditer(t):
        window = t[max(0, m.start() - 25): m.start()]
        if _NEGATION_BEFORE.search(window):
            continue
        return True
    return False


def says_not_emergency(text: str) -> bool:
    return bool(_NOT_EMERGENCY_RE.search(text.lower()))


@dataclass
class Entities:
    date: Optional[str] = None
    ambiguous_date: bool = False
    time_preference: Optional[str] = None
    time: Optional[str] = None
    appointment_id: Optional[str] = None
    type_mentioned: bool = False


def extract_entities(text: str, today: dt.date) -> Entities:
    d = parse_date(text, today)
    return Entities(
        date=d.date,
        ambiguous_date=d.ambiguous,
        time_preference=parse_time_preference(text),
        time=parse_time(text),
        appointment_id=find_appointment_id(text),
        type_mentioned=mentions_consultation_type(text),
    )
