"""Script integration checks using synthetic data, a small backbone and a test logger."""

import io
import os
import runpy
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

import src.dataset as dataset
import vit_foci
from src.model.build import run_tag
from src.utils.mrfp_config import MRFP_VARIANTS
from test_mrfp_variants import TinyBackbone, TinySACBackbone

ROOT = Path(__file__).resolve().parents[1]


class RecordingWriter:
    def __init__(self, *args, **kwargs):
        self.scalars, self.text = [], []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, float(value), step))

    def add_text(self, name, value, *args, **kwargs):
        self.text.append((name, value))

    def add_histogram(self, *args, **kwargs):
        pass

    def add_images(self, *args, **kwargs):
        pass

    def close(self):
        pass


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        old_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, old_cwd)
        self.data = self.root / "data/foci"
        for split in ("train", "val", "test"):
            (self.data / "images" / split).mkdir(parents=True)
            (self.data / "labels" / split).mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (80, 60), (90 + index * 20, 110, 150)).save(
                    self.data / "images" / split / f"{index}.png")
                (self.data / "labels" / split / f"{index}.txt").write_text("0 0.5 0.5 0.3 0.4\n")
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.dict(dataset.DATASETS, {"foci": dict(
            root=self.data, img_size=64, classes=["virus"], splits=("train", "val"))}))
        loader = dataset.build_folder_loaders
        def zero_workers(*args, **kwargs):
            kwargs["num_workers"] = 0
            return loader(*args, **kwargs)
        stack.enter_context(patch.object(dataset, "build_folder_loaders", side_effect=zero_workers))
        stack.enter_context(patch("src.model.build.ViTDetBackbone", TinyBackbone))
        stack.enter_context(patch("src.model.build.ViTDetSACBackbone", TinySACBackbone))
        stack.enter_context(patch("torch.cuda.is_available", return_value=False))
        logger = types.ModuleType("torch.utils.tensorboard")
        logger.SummaryWriter = RecordingWriter
        stack.enter_context(patch.dict(sys.modules, {"torch.utils.tensorboard": logger}))

    def run_script(self, name, args):
        output = io.StringIO()
        with patch.object(sys, "argv", [name, *args]), redirect_stdout(output):
            result = runpy.run_path(str(ROOT / name), run_name="__main__")
        return result, output.getvalue()

    def test_all_variants_train_load_infer_and_aggregate_without_sac(self):
        parameter_counts = {}
        for variant in MRFP_VARIANTS:
            with self.subTest(variant=variant):
                trained, _ = self.run_script("train.py", [
                    "--dataset", "foci", "--aug", "A", "--seed", "42", "--epochs", "1", "--mrfp", variant])
                self.assertIs(type(trained["model"].backbone), TinyBackbone)
                self.assertEqual(trained["sac_mods"], [])
                tag = run_tag("foci", 64, False, "A", 42, mrfp_variant=variant)
                self.assertTrue((self.root / f"best_{tag}.pt").exists())
                self.assertTrue((self.root / f"last_{tag}.pt").exists())
                parameter_counts[variant] = sum(p.numel() for p in trained["model"].parameters())
                self.assertEqual(sum(p.numel() for group in trained["optimizer"].param_groups for p in group["params"]),
                                 parameter_counts[variant])
                inferred, _ = self.run_script("inference.py", [
                    "--dataset", "foci", "--aug", "A", "--seed", "42", "--mrfp", variant,
                    "--conf", "0.99", str(self.data / "images/val/0.png")])
                self.assertEqual(inferred["CKPT"], f"best_{tag}.pt")
                self.assertTrue((inferred["OUT_DIR"] / "0.png").exists())
        evaluated, output = self.run_script("evaluate.py", [
            "--dataset", "foci", "--aug", "A", "--seeds", "42", "--mrfp-variants", *MRFP_VARIANTS])
        self.assertNotIn("[skip]", output)
        self.assertEqual(len(evaluated["out_rows"]), 2 * len(MRFP_VARIANTS))
        self.assertEqual({r["sac_rates"] for r in evaluated["out_rows"]}, {"none"})
        self.assertEqual({r["mrfp_variant"] for r in evaluated["out_rows"]}, set(MRFP_VARIANTS))
        self.assertTrue((self.root / f"{evaluated['stem']}.csv").exists())
        self.assertTrue((self.root / f"{evaluated['stem']}.md").exists())
        self.assertLess(parameter_counts["plain-dw"], parameter_counts["full"])
        self.assertLess(parameter_counts["no-switch"], parameter_counts["full"])
        self.assertLess(parameter_counts["single-dw"], parameter_counts["plain-dw"])
        self.assertLess(parameter_counts["plain-dw"], parameter_counts["plain-dw-sac"])

    def test_training_accepts_existing_three_split_datasets(self):
        with patch.dict(dataset.DATASETS["foci"], {"splits": ("train", "val", "test")}):
            trained, _ = self.run_script("train.py", ["--dataset", "foci", "--epochs", "1", "--mrfp", "full"])
        self.assertEqual(len(trained["train_loader"].dataset), 2)
        self.assertEqual(len(trained["val_loader"].dataset), 2)


class RunnerTests(unittest.TestCase):
    def run_main(self, argv):
        original_cwd = Path.cwd()
        output = io.StringIO()
        with patch.object(sys, "argv", ["vit_foci.py", *argv]), \
             patch.object(vit_foci, "run_cmd") as run, \
             patch.object(vit_foci, "ensure_dataset") as prepare, \
             patch.object(vit_foci, "verify_repo_imports") as verify, redirect_stdout(output):
            try:
                vit_foci.main()
            finally:
                os.chdir(original_cwd)
        return run, prepare, verify, output.getvalue()

    def test_sweep_forwards_every_variant_seed_and_same_backbone_to_evaluation(self):
        variants = ("plain-dw", "plain-dw-sac", "single-dw")
        run, prepare, verify, _ = self.run_main([
            "--skip-prepare", "--seeds", "42", "43", "44", "--mrfp-variants", *variants])
        prepare.assert_not_called()
        verify.assert_called_once()
        calls = [c.args[0] for c in run.call_args_list]
        self.assertEqual(len(calls), 10)
        self.assertTrue(all("--sac-rates" not in cmd for cmd in calls))
        pairs = {(cmd[cmd.index("--mrfp") + 1], cmd[cmd.index("--seed") + 1]) for cmd in calls[:-1]}
        self.assertEqual(pairs, {(v, s) for v in variants for s in ("42", "43", "44")})
        evaluation = calls[-1]
        self.assertEqual(evaluation[evaluation.index("--configs") + 1:evaluation.index("--mrfp-variants")], ["none"])
        self.assertEqual(evaluation[evaluation.index("--mrfp-variants") + 1:], list(variants))

    def test_sac_is_only_used_if_requested_and_is_forwarded_consistently(self):
        run, _, _, _ = self.run_main(["--skip-prepare", "--seeds", "42", "--mrfp", "full", "--sac-rates", "1,3,5"])
        training, evaluation = [c.args[0] for c in run.call_args_list]
        self.assertEqual(training[training.index("--sac-rates") + 1], "1,3,5")
        self.assertEqual(evaluation[evaluation.index("--configs") + 1], "1,3,5")

    def test_old_runner_defaults_are_preserved(self):
        run, _, _, _ = self.run_main(["--skip-prepare", "--seeds", "42"])
        training, evaluation = [c.args[0] for c in run.call_args_list]
        self.assertNotIn("--mrfp", training)
        self.assertNotIn("--mrfp-variants", evaluation)
        self.assertEqual(evaluation[evaluation.index("--configs") + 1:], ["none", "1,3,5", "1"])

    def test_dry_run_does_not_prepare_build_or_spawn_processes(self):
        with patch.object(sys, "argv", ["vit_foci.py", "--dry-run", "--seeds", "42", "--mrfp", "plain-dw"]), \
             patch.object(vit_foci, "ensure_dataset") as prepare, \
             patch.object(vit_foci, "verify_repo_imports") as verify, \
             patch.object(vit_foci.subprocess, "run") as spawn, redirect_stdout(io.StringIO()) as output:
            original_cwd = Path.cwd()
            try:
                vit_foci.main()
            finally:
                os.chdir(original_cwd)
        prepare.assert_not_called()
        verify.assert_not_called()
        spawn.assert_not_called()
        self.assertIn("--mrfp plain-dw", output.getvalue())


if __name__ == "__main__":
    unittest.main()
