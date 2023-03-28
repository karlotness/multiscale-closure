def qg_model_to_args(model):
    qg_args = {
        "nx",
        "ny",
        "L",
        "W",
        "rek",
        "filterfac",
        "f",
        "g",
        "beta",
        "rd",
        "delta",
        "H1",
        "U1",
        "U2",
        "precision",
    }
    return {
        arg: getattr(model, arg) for arg in qg_args
    }
