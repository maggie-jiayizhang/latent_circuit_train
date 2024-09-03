def parse_dict(d, keys):
    vals = []
    for k in keys:
        vals.append(d[k])
    return tuple(vals)