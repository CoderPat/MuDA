from typing import Type, Dict, Any, Callable
import importlib
import os

from muda.tagger import Tagger

TAGGER_REGISTRY: Dict[str, Type[Tagger]] = {}


def register_tagger(tagger_name: str) -> Callable[[Type[Tagger]], None]:
    tagger_name = tagger_name.lower()

    def register_tagger_cls(cls: Type[Tagger]) -> None:
        if tagger_name in TAGGER_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(tagger_name))
        if not issubclass(cls, Tagger):
            raise ValueError(
                "Model ({}: {}) must extend Tagger".format(tagger_name, cls.__name__)
            )

        TAGGER_REGISTRY[tagger_name] = cls

    return register_tagger_cls


def import_taggers(langdir: str, namespace: str) -> None:
    for file in os.listdir(langdir):
        path = os.path.join(langdir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            tagger_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + tagger_name)


def create_tagger(langcode: str, **kwargs: Any) -> Tagger:
    # standardize tagger name by langcode
    tagger_name = f"{langcode}_tagger"
    tagger = TAGGER_REGISTRY[tagger_name](**kwargs)
    return tagger


langdir = os.path.dirname(__file__)
import_taggers(langdir, __name__)
