import importlib

def make_module_factory(module, class_name):

    def make_module():
        mod_name = importlib.import_module(f".{module}", __name__)
        return getattr(mod_name, class_name)()

    return make_module

ARCHITECTURES = {
    "closure-cnn-v1": make_module_factory("cnn", "ClosureCnnV1"),
    "rnn-unet": make_module_factory("rnn_unet", "RNNUNetUV"),
}
