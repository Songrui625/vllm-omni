import torch.nn as nn

from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()


class DummyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.text_encoder = nn.Module()
        self.vae = nn.Module()


# LTX2TwoStagesPipeline-like nested pipeline
class NestedDummyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.pipe = DummyPipeline()
        self.upsample_pipe = DummyPipeline()


class TestModuleDiscovery:
    def test_discover_basic(self):
        pipeline = DummyPipeline()
        modules = ModuleDiscovery.discover(pipeline)
        assert len(modules.dits) > 0
        assert len(modules.encoders) > 0
        assert modules.vae is not None

    def test_discover_nested(self):
        pipeline = NestedDummyPipeline()
        modules = ModuleDiscovery.discover(pipeline)
        assert len(modules.dits) > 0
        assert len(modules.encoders) > 0
        assert modules.vae is not None
