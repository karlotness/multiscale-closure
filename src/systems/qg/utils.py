import json
import pyqg_jax

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


def qg_model_param_json(model):
    args = qg_model_to_args(model)
    args["precision"] = args["precision"].name
    return json.dumps(args)


def qg_model_from_param_json(param):
    args = json.loads(param)
    if "precision" in args:
        args["precision"] = pyqg_jax.state.Precision[args["precision"]]
    args.pop("dt", None)
    args.pop("tmax", None)
    args.pop("tavestart", None)
    args.pop("taveint", None)
    return pyqg_jax.qg_model.QGModel(**args)
