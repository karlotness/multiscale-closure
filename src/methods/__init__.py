import importlib
import dataclasses
import typing
import re


@dataclasses.dataclass
class ModuleFactory:
    module: str
    class_name: str
    fixed_args: None | dict[str, typing.Any] = None

    def __call__(self, *args, **kwargs):
        fixed_args = {} if self.fixed_args is None else self.fixed_args
        module = importlib.import_module(f".{self.module}", __name__)
        return getattr(module, self.class_name)(*args, **kwargs, **fixed_args)

    def __repr__(self):
        if self.fixed_args is not None:
            fixed_str = f", fixed_args={self.fixed_args!r}"
        else:
            fixed_str = ""
        return f"ModuleFactory({__name__}.{self.module}:{self.class_name}{fixed_str})"


ARCHITECTURES = {
    "gz-fcnn-v1": ModuleFactory("gz_fcnn", "GZFCNN"),
    "gz-fcnn-v1-large": ModuleFactory("gz_fcnn", "LargeGZFCNN"),
    "gz-fcnn-v1-medium": ModuleFactory("gz_fcnn", "MediumGZFCNN"),
    "unet-v1": ModuleFactory("basic_unet", "BasicUNetV1"),
}


def get_net_constructor(arch):
    if (constructor := ARCHITECTURES.get(arch)) is not None:
        return constructor
    elif m := re.fullmatch(r"stacked-gz-fcnn-v(?P<version>\d+)-(?:(?P<size>medium)-)?d(?P<depth>\d+)", arch, re.ASCII):
        version = int(m.group("version"))
        size = m.group("size")
        depth = int(m.group("depth"))
        cls_name = (size or "").capitalize() + "StackedGZFCNN" + (f"V{version:d}" if version != 1 else "")
        return ModuleFactory("stacked_gz_fcnn", cls_name, fixed_args={"depth": depth})
    elif m := re.fullmatch(r"shallow-gz-fcnn-v(?P<version>\d+)-(?:(?P<size>\w+)-)?l(?P<layers>\d+)", arch, re.ASCII):
        return ModuleFactory(
            "fcnn_shallow",
            "make_shallow_fcnn",
            fixed_args={
                "arch_version": int(m.group("version")),
                "arch_size": (m.group("size") or "small"),
                "arch_layers": int(m.group("layers")),
            },
        )
    elif m := re.fullmatch(r"stacked-noscale-net-v(?P<version>\d+)-(?P<arch_str>.+)", arch, re.ASCII):
        return ModuleFactory(
            "stacked_noscale_nets",
            "make_stacked_noscale_net",
            fixed_args={
                "arch_version": int(m.group("version")),
                "arch_str": m.group("arch_str"),
            },
        )
    raise ValueError(f"unknown architecture {arch}")
