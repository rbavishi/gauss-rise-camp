def prepare_json_string(string: str):
    return '' + string.replace("'", '\\\\\\"').replace('"', '"') + ''
