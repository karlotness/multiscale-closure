import importlib

def make_module_factory(module, class_name):

    def make_module(*args, **kwargs):
        mod_name = importlib.import_module(f".{module}", __name__)
        return getattr(mod_name, class_name)(*args, **kwargs)

    return make_module


ARCHITECTURES = {
    "gz-fcnn-v1": make_module_factory("gz_fcnn", "GZFCNN"),
    "gz-fcnn-v1-large": make_module_factory("gz_fcnn", "LargeGZFCNN"),
    "gz-fcnn-v1-medium": make_module_factory("gz_fcnn", "MediumGZFCNN"),
}
