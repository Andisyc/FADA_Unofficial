"""Independent sealed APPO training, persistence and playback assembly."""
from unilab.algos.torch.appo.runtime import APPORuntime
from .config import validate_collection_contract


def resolve_runtime(rl_cfg):
    validate_collection_contract(rl_cfg)
    from .runner import OfficialFadaAPPORunner
    from unilab.training.fada_official_appo import run_training
    from .playback import play_checkpoint
    return APPORuntime(OfficialFadaAPPORunner, play_checkpoint, run_training)
