import importlib

def make_module_factory(module, class_name, fixed_args=None):
    if fixed_args is None:
        fixed_args = {}

    def make_module(*args, **kwargs):
        mod_name = importlib.import_module(f".{module}", __name__)
        return getattr(mod_name, class_name)(*args, **kwargs, **fixed_args)

    return make_module


ARCHITECTURES = {
    "gz-fcnn-v1": make_module_factory("gz_fcnn", "GZFCNN"),
    "gz-fcnn-v1-large": make_module_factory("gz_fcnn", "LargeGZFCNN"),
    "gz-fcnn-v1-medium": make_module_factory("gz_fcnn", "MediumGZFCNN"),
    "unet-v1": make_module_factory("basic_unet", "BasicUNetV1"),
    "stacked-gz-fcnn-v1-d2": make_module_factory("stacked_gz_fcnn", "StackedGZFCNN", fixed_args={"depth": 2}),
    "stacked-gz-fcnn-v1-d3": make_module_factory("stacked_gz_fcnn", "StackedGZFCNN", fixed_args={"depth": 3}),
    "stacked-gz-fcnn-v1-medium-d2": make_module_factory("stacked_gz_fcnn", "MediumStackedGZFCNN", fixed_args={"depth": 2}),
    "stacked-gz-fcnn-v1-medium-d3": make_module_factory("stacked_gz_fcnn", "MediumStackedGZFCNN", fixed_args={"depth": 3}),
}
