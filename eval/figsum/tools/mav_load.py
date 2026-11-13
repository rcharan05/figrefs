import typing, collections, torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode, typing.Any,
                                      dict, list, int, collections.defaultdict])
from maverick import Maverick
def load():
    return Maverick(hf_name_or_path="sapienzanlp/maverick-mes-ontonotes", device="cpu")
