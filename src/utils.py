import random
import string
import logging
import subprocess
import codecs
import dataclasses
import os
import pathlib
import contextlib


@contextlib.contextmanager
def rename_save_file_path(path):
    final_path = pathlib.Path(path).resolve()
    rand_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=15))
    target_path = final_path.with_name(f"{final_path.name}.PART{rand_suffix}").resolve()
    with open(target_path, "x") as _target_file:
        pass
    yield str(target_path)
    # Do the final rename
    os.replace(target_path, final_path)


@contextlib.contextmanager
def rename_save_file(file, mode, *args, **kwargs):
    final_path = pathlib.Path(file)
    rand_suffix = random.choices(string.ascii_lowercase + string.digits, k=15)
    target_path = final_path.with_name(f"{final_path.name}.PART{rand_suffix}")
    # Open temporary file
    if mode.lower() not in {"w", "wb", "bw"}:
        raise ValueError("file must be opened for writing")
    with open(target_path, mode.lower().replace("w", "x"), **kwargs) as target_file:
        yield target_file
    # Do the final rename
    os.replace(target_path, final_path)


def check_environment_variables(base_logger=None):
    if base_logger:
        logger = base_logger.getChild("jax_env_check")
    else:
        logger = logging.getLogger("jax_env_check")
    ok = True
    if "JAX_ENABLE_X64" not in os.environ or os.environ["JAX_ENABLE_X64"] != "True":
        # Float64 not enabled
        ok = False
        logger.warning("JAX float64 support is not enabled, this causes numerical issues in QG")
        logger.info("set environment variable JAX_ENABLE_X64=True")
    else:
        logger.info("JAX float64 support enabled")
    if "JAX_DEFAULT_DTYPE_BITS" not in os.environ or os.environ["JAX_DEFAULT_DTYPE_BITS"] != "32":
        # wrong default dtype size
        ok = False
        logger.warning("Default dtype size is not 32bits. This will create float64 constants by default")
        logger.info("set environment variable JAX_DEFAULT_DTYPE_BITS=32")
    else:
        logger.info("JAX dtypes defaulting to 32 bits")
    return ok


def set_up_logging(level="info", out_file=None):
    num_level = getattr(logging, level.upper(), None)
    if not isinstance(num_level, int):
        raise ValueError("Invalid log level: {}".format(level))
    handlers = []
    handlers.append(logging.StreamHandler())
    if out_file:
        handlers.append(logging.FileHandler(filename=out_file, encoding="utf8"))
    logging.basicConfig(level=num_level, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@dataclasses.dataclass
class CommitInfo:
    hash: str
    clean_worktree: bool


def get_git_info(base_logger=None):
    git_dir = pathlib.Path(__file__).parent
    if base_logger:
        logger = base_logger.getChild("gitinfo")
    else:
        logger = logging.getLogger("gitinfo")
    try:
        commit_id_out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True, cwd=git_dir)
        commit_id = codecs.decode(commit_id_out.stdout).strip()
        clean_tree_out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, check=True, cwd=git_dir)
        clean_worktree = len(clean_tree_out.stdout) == 0
        return CommitInfo(hash=commit_id, clean_worktree=clean_worktree)
    except Exception:
        logger.exception("Failed to get information on git status")
        return None


@dataclasses.dataclass
class ModuleInfo:
    name: str
    version: str
    raw_name: str


def get_loaded_modules(base_logger=None):
    if base_logger:
        logger = base_logger.getChild("moduleinfo")
    else:
        logger = logging.getLogger("moduleinfo")

    module_list = os.environ.get("LOADEDMODULES", None)
    if module_list is None:
        logger.warning("Failed to get linux module info")
        return []
    modules = []
    for raw_name in module_list.strip().split(":"):
        # Split components of module name
        if "/" in raw_name:
            components = raw_name.split("/")
        else:
            components = raw_name.split("-")
        # Extract components
        name = components[0]
        if len(components) > 1:
            version = components[-1]
        else:
            version = ""
        modules.append(ModuleInfo(name=name, version=version,
                                  raw_name=raw_name))
    return modules
