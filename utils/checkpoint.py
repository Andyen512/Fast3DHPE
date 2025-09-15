def load_state_flexible(model, state_dict):
    msd = model.state_dict()
    fixed = {}
    for k, v in state_dict.items():
        kk = k
        for pref in ("module.", "core."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        if kk in msd and msd[kk].shape == v.shape:
            fixed[kk] = v
    missing = [k for k in msd.keys() if k not in fixed]
    model.load_state_dict({**msd, **fixed}, strict=False)
    return {"loaded": list(fixed.keys()), "missing": missing}
