"""Offline CPU checks: python -m unittest discover -s tests -v."""

import argparse
import copy
import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from src.model.mrfp import MRFP, MRFPNeck, CoMerMRFP, SwitchableDepthwiseConv
from src.model.build import build_model, run_tag
from src.model.fpn import SimpleFeaturePyramid
from src.losses.detection_loss import DetectionLoss
from src.utils.mrfp_config import MRFP_VARIANTS, add_mrfp_args, requested_variants, resolve_mrfp

torch.set_num_threads(2)


class TinyBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embed_dim = 32
        self.proj = nn.Conv2d(3, 32, 16, stride=16)

    def forward(self, x):
        return self.proj(x)


class TinySACBackbone(TinyBackbone):
    pass


class OriginalMRFP(nn.Module):
    """Frozen pre-ablation architecture, for initialization/gradient regression checks."""
    def __init__(self, ch):
        super().__init__()
        hidden = ch // 2
        self.reduce = nn.Conv2d(ch, hidden, 1)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=1, dilation=1)
        self.conv_dilated = nn.Conv2d(hidden, hidden, 3, padding=3, dilation=3)
        self.switch = nn.Conv2d(hidden, 2, 1)
        nn.init.zeros_(self.switch.weight)
        nn.init.zeros_(self.switch.bias)
        self.expand = nn.Conv2d(hidden, ch, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        z = F.gelu(self.reduce(x))
        a, b = self.conv3(z), self.conv_dilated(z)
        gates = torch.softmax(self.switch(F.avg_pool2d(z, 5, stride=1, padding=2)), dim=1)
        return x + self.gamma * self.expand(F.gelu(gates[:, :1] * a + gates[:, 1:] * b))


class MRFPTests(unittest.TestCase):
    def test_full_matches_original_weights_rng_outputs_and_gradients(self):
        torch.manual_seed(42)
        old = OriginalMRFP(32)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        new = MRFP(32)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        x = torch.randn(2, 32, 10, 12)
        a, b = old(x), new(x)
        self.assertTrue(torch.equal(a, b))
        a.square().mean().backward()
        b.square().mean().backward()
        for p, q in zip(old.parameters(), new.parameters()):
            self.assertTrue(torch.equal(p.grad, q.grad))

    def test_controls_remove_exactly_the_requested_components(self):
        for variant in ("full", "no-dilation", "no-switch", "no-dilation-no-switch"):
            with self.subTest(variant=variant):
                block = MRFP(32, variant)
                dilation = 1 if "no-dilation" in variant else 3
                self.assertEqual(block.conv_dilated.dilation, (dilation, dilation))
                self.assertEqual(block.conv_dilated.padding, (dilation, dilation))
                self.assertEqual(block.conv3.groups, 1)
                self.assertEqual(block.conv_dilated.groups, 1)
                self.assertEqual(block.switch is None, "no-switch" in variant)
                self.assertEqual(any(n.startswith("switch.") for n, _ in block.named_parameters()),
                                 "no-switch" not in variant)
                x = torch.randn(2, 32, 7, 9)
                self.assertEqual(block(x).shape, x.shape)
                with torch.no_grad():
                    block.gamma.zero_()
                self.assertTrue(torch.equal(block(x), x))

    def test_no_switch_matches_uniform_gates_with_common_weights(self):
        full, fixed = MRFP(32), MRFP(32, "no-switch")
        fixed.load_state_dict({k: v for k, v in full.state_dict().items() if not k.startswith("switch.")})
        x = torch.randn(2, 32, 7, 9)
        self.assertTrue(torch.equal(full(x), fixed(x)))

    def test_comer_kernel_split_level_parameters_and_reference_structure(self):
        block = CoMerMRFP(32, 3)
        self.assertFalse(hasattr(block, "gamma"))
        self.assertFalse(hasattr(block, "switch"))
        self.assertEqual(block.reduce.out_channels, 16)
        for level in block.levels:
            for name, kernel in (("dw3", 3), ("dw5", 5)):
                conv = level[name]
                self.assertEqual(conv.kernel_size, (kernel, kernel))
                self.assertEqual(conv.dilation, (1, 1))
                self.assertEqual(conv.groups, 8)
                self.assertEqual(conv.in_channels, 8)
            self.assertIsInstance(level["norm"], nn.BatchNorm2d)
        self.assertEqual(len({id(level["dw3"].weight) for level in block.levels}), 3)
        features = [torch.randn(2, 32, size, size) for size in (16, 8, 4)]
        results = block(features)
        self.assertEqual([t.shape for t in results], [t.shape for t in features])
        sum(x.square().mean() for x in results).backward()
        for p in block.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        for level in block.levels:
            self.assertEqual(level["norm"].num_batches_tracked.item(), 1)

    def test_comer_map_forward_matches_token_fc_reference(self):
        # Test the map adapter against the official operation order with identical
        # weights: token FC -> split DW3/DW5 -> concat -> BN/GELU -> GELU -> token FC.
        block = CoMerMRFP(32, 3).eval()
        features = [torch.randn(2, 32, size, size) for size in (16, 8, 4)]
        actual = block(features)
        for x, level, output in zip(features, block.levels, actual):
            tokens = x.permute(0, 2, 3, 1)
            reduced = F.linear(tokens, block.reduce.weight[:, :, 0, 0], block.reduce.bias)
            a, b = reduced.permute(0, 3, 1, 2).chunk(2, dim=1)
            mixed = torch.cat((level["dw3"](a), level["dw5"](b)), dim=1)
            mixed = F.gelu(F.gelu(level["norm"](mixed)))
            expected = F.linear(mixed.permute(0, 2, 3, 1),
                                block.expand.weight[:, :, 0, 0], block.expand.bias).permute(0, 3, 1, 2)
            torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_single_dw_matches_multikernel_with_outer_5x5_weights_zeroed(self):
        # A single DW over all channels equals the split reference when the 5x5
        # outer ring is zero. This checks fusion, channel order and normalization.
        single = CoMerMRFP(32, 3, variant="single-dw").eval()
        multi = CoMerMRFP(32, 3).eval()
        multi.reduce.load_state_dict(single.reduce.state_dict())
        multi.expand.load_state_dict(single.expand.state_dict())
        for a, b in zip(single.levels, multi.levels):
            conv = a["dw"]
            self.assertEqual(conv.kernel_size, (3, 3))
            self.assertEqual(conv.dilation, (1, 1))
            self.assertEqual(conv.groups, 16)
            self.assertEqual(list(a), ["dw", "norm"])
            b["norm"].load_state_dict(a["norm"].state_dict())
            with torch.no_grad():
                b["dw3"].weight.copy_(conv.weight[:8])
                b["dw3"].bias.copy_(conv.bias[:8])
                b["dw5"].weight.zero_()
                b["dw5"].weight[:, :, 1:4, 1:4].copy_(conv.weight[8:])
                b["dw5"].bias.copy_(conv.bias[8:])
        features = [torch.randn(2, 32, h, w) for h, w in ((16, 20), (8, 10), (4, 5))]
        for actual, expected in zip(single(features), multi(features)):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        params = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(params(multi) - params(single), 3 * 16 * 8)

    def test_neck_sac_has_learnable_switches_for_both_kernel_groups(self):
        block = CoMerMRFP(32, 3, variant="plain-dw-sac")
        self.assertFalse(hasattr(block, "gamma"))
        for level in block.levels:
            for name, kernel in (("dw3", 3), ("dw5", 5)):
                sac = level[name]
                self.assertIsInstance(sac, SwitchableDepthwiseConv)
                for branch, rate in zip(sac.branches, (1, 3)):
                    self.assertEqual(branch.kernel_size, (kernel, kernel))
                    self.assertEqual(branch.dilation, (rate, rate))
                    self.assertEqual(branch.padding, ((kernel // 2) * rate,) * 2)
                    self.assertEqual(branch.groups, 8)
                self.assertEqual(torch.count_nonzero(sac.switch.weight).item(), 0)
                self.assertEqual(torch.count_nonzero(sac.switch.bias).item(), 0)
        features = [torch.randn(2, 32, h, w) for h, w in ((16, 20), (8, 10), (4, 5))]
        outputs = block(features)
        self.assertEqual([o.shape for o in outputs], [f.shape for f in features])
        sum(output.square().mean() for output in outputs).backward()
        for name, p in block.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        for level in block.levels:
            for key in ("dw3", "dw5"):
                self.assertGreater(level[key].switch.weight.grad.abs().max().item(), 0)

    def test_neck_sac_uniform_initialization_and_branch_selection(self):
        x = torch.randn(2, 8, 11, 13)
        for kernel in (3, 5):
            sac = SwitchableDepthwiseConv(8, kernel)
            expected = .5 * sac.branches[0](x) + .5 * sac.branches[1](x)
            torch.testing.assert_close(sac(x), expected, rtol=0, atol=0)
            for index in (0, 1):
                with torch.no_grad():
                    sac.switch.bias.fill_(-20)
                    sac.switch.bias[index] = 20
                torch.testing.assert_close(sac(x), sac.branches[index](x), atol=1e-6, rtol=1e-5)

    def test_invalid_variants_and_comer_channel_counts_fail(self):
        with self.assertRaises(ValueError):
            MRFP(32, "typo")
        with self.assertRaises(ValueError):
            CoMerMRFP(30)
        with self.assertRaises(ValueError):
            CoMerMRFP(32, variant="typo")
        with self.assertRaises(ValueError):
            CoMerMRFP(32, 3)([torch.zeros(1, 32, 8, 8)])


@patch("src.model.build.ViTDetBackbone", TinyBackbone)
@patch("src.model.build.ViTDetSACBackbone", TinySACBackbone)
class ModelTests(unittest.TestCase):
    def test_legacy_pairing_and_new_independent_neck(self):
        plain = build_model(64, 2, out_ch=32)
        legacy = build_model(64, 2, out_ch=32, sac_rates=(1, 3, 5))
        self.assertIsInstance(plain.neck, SimpleFeaturePyramid)
        self.assertIsInstance(legacy.neck, MRFPNeck)
        self.assertIsInstance(legacy.backbone, TinySACBackbone)
        for variant in MRFP_VARIANTS:
            model = build_model(64, 2, out_ch=32, mrfp_variant=variant)
            self.assertIs(type(model.backbone), TinyBackbone)
            self.assertEqual(isinstance(model.neck, MRFPNeck), variant != "none")
        sac_only = build_model(64, 2, out_ch=32, sac_rates=(1,), mrfp_variant="none")
        self.assertIsInstance(sac_only.neck, SimpleFeaturePyramid)
        self.assertIsInstance(sac_only.backbone, TinySACBackbone)

    def test_explicit_full_and_legacy_sac_are_identical(self):
        torch.manual_seed(42)
        legacy = build_model(64, 2, out_ch=32, sac_rates=(1, 3, 5))
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        explicit = build_model(64, 2, out_ch=32, sac_rates=(1, 3, 5), mrfp_variant="full")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]))

    def test_all_variants_training_empty_targets_and_checkpoint_roundtrip(self):
        for variant in MRFP_VARIANTS:
            for use_p2 in (False, True):
                with self.subTest(variant=variant, use_p2=use_p2):
                    model = build_model(64, 2, out_ch=32, use_p2=use_p2, mrfp_variant=variant)
                    criterion = DetectionLoss(model)
                    x = torch.randn(2, 3, 64, 64)
                    for empty in (False, True):
                        model.zero_grad()
                        target = [torch.empty(0, 5), torch.empty(0, 5)] if empty else [
                            torch.tensor([[0., .5, .5, .5, .5]]), torch.empty(0, 5)]
                        loss, items = criterion(model(x), target, "cpu")
                        self.assertEqual(items.shape, (3,))
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        for p in model.parameters():
                            if p.grad is not None:
                                self.assertTrue(torch.isfinite(p.grad).all())
                    buffer = io.BytesIO()
                    torch.save(model.state_dict(), buffer)
                    buffer.seek(0)
                    restored = build_model(64, 2, out_ch=32, use_p2=use_p2, mrfp_variant=variant)
                    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
                    model.eval()
                    restored.eval()
                    with torch.no_grad():
                        self.assertTrue(torch.equal(model(x)[0], restored(x)[0]))

    def test_autocast_gradients_for_each_enabled_variant(self):
        for variant in MRFP_VARIANTS[1:]:
            with self.subTest(variant=variant):
                model = build_model(64, 2, out_ch=32, mrfp_variant=variant)
                with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                    loss, _ = DetectionLoss(model)(model(torch.randn(2, 3, 64, 64)),
                                                   [torch.tensor([[0., .5, .5, .5, .5]]), torch.empty(0, 5)], "cpu")
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for p in model.parameters():
                    if p.grad is not None:
                        self.assertTrue(torch.isfinite(p.grad).all())


class NamingTests(unittest.TestCase):
    def test_legacy_tags_and_distinct_new_names(self):
        self.assertEqual(run_tag("foci", 640, False, "A", 42), "foci_640_p3p5_augA_s42")
        self.assertEqual(run_tag("foci", 640, False, "A", 42, sac=(1, 3, 5)),
                         "foci_640_p3p5_augA_s42_sac135")
        tags = {run_tag("foci", 640, False, "A", 42, mrfp_variant=v) for v in MRFP_VARIANTS}
        self.assertEqual(len(tags), len(MRFP_VARIANTS))
        self.assertTrue(all("_mrfp-" in tag for tag in tags))

    def test_cli_defaults_single_choice_sweep_and_invalid_inputs(self):
        parser = argparse.ArgumentParser()
        add_mrfp_args(parser, sweep=True)
        self.assertEqual(requested_variants(parser.parse_args([])), [None])
        self.assertEqual(requested_variants(parser.parse_args(["--mrfp"])), ["full"])
        self.assertEqual(requested_variants(parser.parse_args(["--mrfp-variants", "none", "full", "full"])),
                         ["none", "full"])
        for args in (["--mrfp", "unknown"], ["--mrfp", "full", "--mrfp-variants", "none"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(args)
        self.assertEqual(resolve_mrfp(None, None), "none")
        self.assertEqual(resolve_mrfp(None, (1, 3, 5)), "full")


if __name__ == "__main__":
    unittest.main()
