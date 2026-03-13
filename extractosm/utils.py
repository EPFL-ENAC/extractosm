import pandas as pd
import re


def _parse_osm_tags(tag_str):
    if pd.isna(tag_str) or tag_str == "":
        return {}
    d = {}
    # Split by comma, then split by =>
    # handle quoted strings
    for kv in re.findall(r'"([^"]+)"=>"(.*?)"', tag_str):
        key, value = kv
        d[key] = value
    return d
