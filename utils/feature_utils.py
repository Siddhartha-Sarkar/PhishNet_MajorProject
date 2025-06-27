import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# utils/feature_utils.py

import re
import numpy as np
import urllib.parse as ul

# -------- Helper: resilient urlparse --------
def safe_urlparse(url: str) -> ul.ParseResult:
    """
    Wrapper around urllib.parse.urlparse that never raises.
    If urlparse() fails (e.g. 'Invalid IPv6 URL'), we try to
    strip bracketed parts or, as a last resort, treat the entire
    string as a path so the rest of the code can continue.
    """
    try:
        return ul.urlparse(url)
    except ValueError:
        cleaned = re.sub(r"\[.*?\]", "", url)
        try:
            return ul.urlparse(cleaned)
        except ValueError:
            dummy = ul.urlparse("")
            return dummy._replace(path=cleaned)

# -------- Suspicious tokens & special chars --------
SUSPICIOUS_WORDS = {
    "secure", "login", "signin", "verify", "update",
    "account", "bank", "paypal", "password", "confirm"
}
SPECIAL_CHARS = set("~!#$%^&*+={}[]|\\;:'\",<>?")

# -------- Feature extractor --------
def lexical_features(url: str) -> dict:
    """
    Extracts a set of lexical features from the given URL string.
    Returns a dict mapping feature names to numeric values.
    """
    url = str(url)
    parsed = safe_urlparse(url)
    host = parsed.netloc or ""
    path = parsed.path
    query = parsed.query
    path_q = path + ("?" + query if query else "")

    # Subdomain parts (all but last two labels)
    subdomains = host.split(".")[:-2] if host.count(".") >= 2 else []
    # First directory in path
    first_dir = path.split("/")[1] if "/" in path[1:] else ""

    # Base counts and lengths
    url_len       = len(url)
    host_len      = len(host)
    path_len      = len(path_q)
    tld_len       = len(host.split(".")[-1]) if "." in host else 0
    first_dir_len = len(first_dir)

    # Character counts
    count_dots            = url.count(".")
    count_hyphens         = url.count("-")
    count_underscores     = url.count("_")
    count_digits          = sum(ch.isdigit() for ch in url)
    count_subdomains      = len(subdomains)
    count_query_params    = query.count("&") + bool(query)
    count_special_chars   = sum(ch in SPECIAL_CHARS for ch in url)

    # Ratios
    digits_ratio  = count_digits / url_len if url_len else 0
    letters_ratio = sum(ch.isalpha() for ch in url) / url_len if url_len else 0

    # Flags
    has_https_scheme        = int(parsed.scheme == "https")
    has_https_token         = int("https" in host.lower())
    has_ip_host             = int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host)))
    has_prefix_suffix       = int("-" in host.split(".")[0]) if host else 0
    has_double_slash_in_path = int("//" in path[1:])
    has_suspicious_tld      = int(host.endswith((
        ".xyz", ".top", ".club", ".info", ".live", ".fit", ".buzz"
    )))
    contains_suspicious_word = int(any(w in url.lower() for w in SUSPICIOUS_WORDS))

    # Shannon entropy
    entropy = 0
    if url_len:
        probs = [url.count(c) / url_len for c in set(url)]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

    return {
        "url_len": url_len,
        "host_len": host_len,
        "path_len": path_len,
        "tld_len": tld_len,
        "first_dir_len": first_dir_len,
        "count_dots": count_dots,
        "count_hyphens": count_hyphens,
        "count_underscores": count_underscores,
        "count_digits": count_digits,
        "count_subdomains": count_subdomains,
        "count_query_params": count_query_params,
        "count_special_chars": count_special_chars,
        "digits_ratio": digits_ratio,
        "letters_ratio": letters_ratio,
        "has_https_scheme": has_https_scheme,
        "has_https_token": has_https_token,
        "has_ip_host": has_ip_host,
        "has_prefix_suffix": has_prefix_suffix,
        "has_double_slash_in_path": has_double_slash_in_path,
        "has_suspicious_tld": has_suspicious_tld,
        "contains_suspicious_word": contains_suspicious_word,
        "entropy": entropy
    }