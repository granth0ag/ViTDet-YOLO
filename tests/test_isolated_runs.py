import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import run_isolated
from src.utils.run_config import run_name


class IsolatedRunnerTests(unittest.TestCase):
    def test_cell_plan_contains_only_original_neck_and_plain_backbone(self):
        cmds = run_isolated.commands("cell-notebook", [42], 200, "test")
        self.assertEqual(len(cmds), 3)
        self.assertEqual([c[c.index("--mrfp") + 1] for c in cmds[:-1]], ["none", "full"])
        for cmd in cmds[:-1]:
            self.assertNotIn("--sac-rates", cmd)
            self.assertEqual(cmd[cmd.index("--lr-backbone") + 1], "1e-4")
            self.assertEqual(cmd[cmd.index("--num-workers") + 1], "2")
        self.assertIn("--full-precision", cmds[-1])
        self.assertEqual(cmds[-1][cmds[-1].index("--conf") + 1], "0.01")

    def test_foci_plan_preserves_current_settings_and_all_pairs(self):
        cmds = run_isolated.commands("foci-current", [42, 43, 44], 200, "test")
        self.assertEqual(len(cmds), 13)
        pairs = {(c[c.index("--mrfp") + 1], c[c.index("--seed") + 1]) for c in cmds[:-1]}
        self.assertEqual(len(pairs), 12)
        for cmd in cmds[:-1]:
            self.assertEqual(cmd[cmd.index("--lr-backbone") + 1], "3e-4")
            self.assertEqual(cmd[cmd.index("--num-workers") + 1], "4")
        self.assertNotIn("--full-precision", cmds[-1])
        self.assertEqual(len(run_isolated.commands("cell-notebook", [42], 200, "test", True)), 1)

    def test_dry_run_has_no_subprocess_side_effects(self):
        with patch.object(sys, "argv", ["run_isolated.py", "--profile", "cell-notebook", "--dry-run"]), \
             patch.object(run_isolated.subprocess, "run") as run, redirect_stdout(io.StringIO()) as output:
            run_isolated.main()
        run.assert_not_called()
        self.assertIn("cell-notebook-e200", output.getvalue())

    def test_names_cannot_escape_output_directory(self):
        import argparse
        for value in ("../x", "a/b", "a b"):
            with self.assertRaises(argparse.ArgumentTypeError):
                run_name(value)
