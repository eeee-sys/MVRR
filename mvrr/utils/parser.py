
import re
import json


def parse_span(span, duration, min_len=-1):
    s, e = span
    s, e = min(duration, max(0, s)), min(duration, max(0, e))
    s, e = min(s, e), max(s, e)

    if min_len != -1 and e - s < min_len:
        h = min_len / 2
        c = min(duration - h, max(h, (s + e) / 2))
        s, e = c - h, c + h

    s, e = min(duration, max(0, s)), min(duration, max(0, e))
    return s, e


def parse_query(query):
    return re.sub(r'\s+', ' ', query).strip().strip('.').strip()


def parse_question(question):
    return re.sub(r'\s+', ' ', question).strip()


def parse_reviser_response(response: str) -> str:
    try:
        data = json.loads(response.strip())
        verdict = data.get("verdict", "").strip()
        if verdict in ("Yes", "No"):
            return verdict
        else:
            raise ValueError(f"Unexpected verdict value: {verdict}")
    except Exception as e:
        raise ValueError(f"Failed to parse reviser response: {response}") from e



def parse_rewriter_response(response: str):
    if not response:
        return "[Empty response]"

    resp = response.strip()

    try:
        data = json.loads(resp)
        if "rewritten_question" in data:
            return data["rewritten_question"]
    except json.JSONDecodeError:
        pass

    match = re.search(r'"rewritten_question"\s*:\s*"([^"]+)"', resp)
    if match:
        return match.group(1)

    return resp