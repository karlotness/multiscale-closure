import importlib

def make_module_factory(module, class_name):

    def make_module(*args, **kwargs):
        mod_name = importlib.import_module(f".{module}", __name__)
        return getattr(mod_name, class_name)(*args, **kwargs)

    return make_module


def make_sdegm_model(model_name, register_module):

    def make_module(*args, **kwargs):
        # Need to import register_module for side effects
        _model_module = importlib.import_module(f".score_sde.models.{register_module}", __name__)
        sdegm_model_utils = importlib.import_module(".score_sde.models.utils", __name__)
        model = sdegm_model_utils.get_model(model_name)
        return model(*args, **kwargs)

    return make_module


ARCHITECTURES = {
    "closure-cnn-v1": make_module_factory("cnn", "ClosureCnnV1"),
    "rnn-unet": make_module_factory("rnn_unet", "RNNUNetUV"),
    "gz-fcnn-v1": make_module_factory("gz_fcnn", "GZFCNNV1"),
    "sdegm-ncsnpp": make_sdegm_model("ncsnpp", "ncsnpp"),
}
