from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import re


def _bbox_from_prov(prov: Dict[str, Any]) -> Optional[Dict[str, float]]:
    bbox = prov.get("bbox") or {}
    try:
        l = float(bbox.get("l", 0.0))
        r = float(bbox.get("r", 0.0))
        b = float(bbox.get("b", 0.0))
        t = float(bbox.get("t", 0.0))
    except (TypeError, ValueError):
        return None
    # basic sanity
    if (r - l) <= 0 or (t - b) <= 0:
        return None
    return {"l": l, "r": r, "b": b, "t": t}


def _is_junk_token(s: str) -> bool:
    s = s.strip()
    if not s:
        return True

    # Drop markdown table artifacts (should be rare in dict mode)
    if s.startswith("|") and s.endswith("|"):
        return True

    # Drop single-char punctuation junk (keeps "s" and "5" etc.)
    if len(s) == 1 and not s.isalnum():
        return True

    return False


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    mid = len(vs) // 2
    if len(vs) % 2 == 1:
        return vs[mid]
    return (vs[mid - 1] + vs[mid]) / 2.0


def docling_dict_to_plain_text(payload: Dict[str, Any]) -> str:
    """
    Build plain text by reconstructing lines from Docling 'texts' using bbox geometry.
    Assumes coord_origin = BOTTOMLEFT (Docling default), so higher on page => larger 't'.
    """
    texts: List[Dict[str, Any]] = payload.get("texts", []) or []

    # Collect tokens
    # token: (page_no, y_center, x_left, height, text)
    tokens: List[Tuple[int, float, float, float, str]] = []
    heights: List[float] = []

    for t in texts:
        s = (t.get("text") or "").strip()
        if _is_junk_token(s):
            continue

        prov_list = t.get("prov") or []
        if not prov_list:
            # If no provenance, skip; you can also append at end if you want.
            continue

        prov0 = prov_list[0]
        page_no = int(prov0.get("page_no", 0))

        bbox = _bbox_from_prov(prov0)
        if not bbox:
            continue

        y_center = (bbox["t"] + bbox["b"]) / 2.0
        x_left = bbox["l"]
        height = (bbox["t"] - bbox["b"])

        tokens.append((page_no, y_center, x_left, height, s))
        heights.append(height)

    if not tokens:
        return ""

    # Sort tokens: page asc, y desc (top->bottom), x asc (left->right)
    tokens.sort(key=lambda z: (z[0], -z[1], z[2]))

    # Adaptive tolerances based on token height
    med_h = _median(heights)
    # y tolerance for same line (tweakable)
    y_tol = max(2.0, med_h * 0.60)
    # vertical gap that triggers a blank line (tweakable)
    gap_tol = max(6.0, med_h * 1.80)

    lines_out: List[str] = []
    current_page: Optional[int] = None

    # Line accumulator
    line_tokens: List[Tuple[float, str]] = []  # (x, text)
    line_y: Optional[float] = None
    last_line_y: Optional[float] = None

    def flush_line():
        nonlocal line_tokens, line_y, last_line_y

        if not line_tokens:
            line_y = None
            return

        # Order tokens in the line by x (already mostly sorted, but safe)
        line_tokens.sort(key=lambda p: p[0])
        line_text = " ".join(tok for _, tok in line_tokens)

        # Clean up spacing around punctuation a tiny bit (safe-ish)
        line_text = re.sub(r"\s+([,.;:])", r"\1", line_text)
        line_text = re.sub(r"\(\s+", "(", line_text)
        line_text = re.sub(r"\s+\)", ")", line_text)

        # Paragraph-ish separation: if the vertical jump is large, add a blank line
        if last_line_y is not None and line_y is not None:
            if (last_line_y - line_y) > gap_tol:
                if lines_out and lines_out[-1].strip() != "":
                    lines_out.append("")

        lines_out.append(line_text)

        last_line_y = line_y
        line_tokens = []
        line_y = None

    for page_no, y, x, _h, s in tokens:
        # Page break handling
        if current_page is None:
            current_page = page_no
        elif page_no != current_page:
            flush_line()
            lines_out.append("")
            lines_out.append(f"--- PAGE {page_no} ---")
            lines_out.append("")
            current_page = page_no
            last_line_y = None  # reset paragraph gap logic per page

        if line_y is None:
            # start new line
            line_y = y
            line_tokens.append((x, s))
            continue

        # Same line if y is close enough
        if abs(y - line_y) <= y_tol:
            line_tokens.append((x, s))
        else:
            # flush current line, start new
            flush_line()
            line_y = y
            line_tokens.append((x, s))

    flush_line()

    # Collapse repeated blank lines
    final: List[str] = []
    blank = False
    for ln in lines_out:
        if ln.strip() == "":
            if not blank:
                final.append("")
            blank = True
        else:
            final.append(ln)
            blank = False

    return "\n".join(final).strip() + "\n"
