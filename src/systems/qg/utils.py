import json
import dataclasses
import jax
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


def qg_model_from_param_json(param, force_single=True):
    args = json.loads(param)
    if force_single and args.get("precision", "DOUBLE").upper() == "DOUBLE":
        args["precision"] = "SINGLE"
    if "precision" in args:
        args["precision"] = pyqg_jax.state.Precision[args["precision"]]
    args.pop("dt", None)
    args.pop("tmax", None)
    args.pop("tavestart", None)
    args.pop("taveint", None)
    return pyqg_jax.qg_model.QGModel(**args)

def register_pytree_dataclass(cls):
    fields = tuple(f.name for f in dataclasses.fields(cls))

    def flatten(obj):
        return [getattr(obj, name) for name in fields], None

    def unflatten(aux_data, flat_contents):
        return cls(**dict(zip(fields, flat_contents, strict=True)))

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls
