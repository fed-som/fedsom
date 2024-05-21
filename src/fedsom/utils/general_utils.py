

def dict_2_string(dict_):
    string = ""
    for k, v in dict_.items():
        if isinstance(v, int):
            string += f"{k}_{v}__"
        else:
            string += f"{k}_{v:.6}__"
    return string[:-2]
