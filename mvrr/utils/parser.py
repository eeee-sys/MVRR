
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


# def parse_reviser_response(response: str):
#     """
#     解析 Reviser 输出，返回 ('Correct',) 或 ('Wrong', '<rewritten_question>').
#
#     参数:
#         response (str): 模型生成的原始字符串
#
#     返回:
#         tuple: ('Correct',) 或 ('Wrong', '<rewritten_question>')
#     """
#     if not response:
#         return ("Wrong", "[Empty response]")
#
#     resp = response.strip()
#
#     # 匹配 Correct
#     if resp.lower().startswith("correct"):
#         return ("Correct",)
#
#     # 匹配 Wrong + 重写问题
#     match = re.match(r"wrong[:：]?\s*(.*)", resp, re.IGNORECASE)
#     if match:
#         rewritten = match.group(1).strip()
#         if not rewritten:
#             rewritten = "[No rewritten question provided]"
#         return ("Wrong", rewritten)
#
#     # 如果不符合任何格式，兜底
#     return ("Wrong", f"[Unrecognized format: {resp}]")

def parse_reviser_response(response: str) -> str:
    """
    解析 REVISER 的输出，提取 verdict 的值。

    参数:
        response (str): 模型生成的字符串（JSON 格式）

    返回:
        str: 'Yes' 或 'No'
    """
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
    """
    解析 Rewriter 输出，提取 rewritten_question 的值。

    参数:
        response (str): 模型生成的原始字符串

    返回:
        str: 提取到的 rewritten_question，若失败则返回原始 response
    """
    if not response:
        return "[Empty response]"

    resp = response.strip()

    # 尝试直接解析 JSON
    try:
        data = json.loads(resp)
        if "rewritten_question" in data:
            return data["rewritten_question"]
    except json.JSONDecodeError:
        pass

    # 如果不是严格 JSON，尝试用正则兜底
    match = re.search(r'"rewritten_question"\s*:\s*"([^"]+)"', resp)
    if match:
        return match.group(1)

    # 实在没法解析，就原样返回
    return resp