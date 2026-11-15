"""CPU checks with synthetic features only; no experiment scores are produced."""

from pathlib import Path
import tempfile, copy, json, hashlib
import numpy as np, torch
from csrd_emt import training as run


def fixtures():
    labels = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
    raw = {}
    expected = {}
    for subject in range(1, 16):
        for session in range(1, 4):
            x = (
                np.random.default_rng(subject * 10 + session)
                .normal(size=(15 * 37, 310))
                .astype("float64")
            )
            raw[subject, session] = dict(
                x=x, y=np.repeat(labels, 37), trial=np.repeat(np.arange(1, 16), 37)
            )
            key = f"S{subject:02}_E{session}"
            expected[key] = {"feature_sha256": hashlib.sha256(x.tobytes()).hexdigest()}
    return raw, expected


def main():
    torch.set_num_threads(2)
    raw, expected = fixtures()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        calls = []

        def load(path, sessions=(1, 2, 3), excluded_subjects=()):
            chosen = {
                k: dict(v) for k, v in raw.items() if k[0] not in excluded_subjects
            }
            subjects = sorted({s for s, e in chosen})
            calls.append(subjects)
            if subjects == [1]:
                assert (root / "fold_01/teacher/best.pt").exists()
                assert (root / "fold_01/student/best.pt").exists()
            manifest = {f"S{s:02}_E{e}": expected[f"S{s:02}_E{e}"] for s, e in chosen}
            return chosen, manifest

        run.load_de = load
        run.MAX_STEPS = 1
        run.VALIDATE_EVERY = 1
        run.DEVICE = "cpu"
        result = run.run_fold(1, "synthetic", root, expected)
        assert len(calls) == 2 and 1 not in calls[0] and calls[1] == [1]
        receipt = run.verify_receipt(root / "fold_01")
        assert receipt["target"] == 1
        pred = np.load(root / "fold_01/target_predictions.npz")
        assert (
            abs(run.balanced_accuracy_score(pred["y"], pred["pred"]) - result["ba"])
            < 1e-12
        )
        assert (
            abs(
                run.f1_score(pred["y"], pred["pred"], average="macro")
                - result["macro_f1"]
            )
            < 1e-12
        )
        before = len(calls)
        run.run_fold(1, "synthetic", root, expected)
        assert len(calls) == before
        print(
            "PASS: source/target isolation, model export, prediction metrics, artifact hashes and completed-fold reuse."
        )
        # Preserve exact stochastic updates across an interruption at validation.
        records = {k: dict(v, x=v["x"].astype("float32")) for k, v in raw.items()}
        tr = run.Data(records, [2, 3, 4, 5])
        va = run.Data(records, [6])
        run.MAX_STEPS = 4
        run.VALIDATE_EVERY = 2
        run.DEADLINE = float("inf")
        cfg = dict(lr=3e-4, lambda_s=0.1, lambda_d=0.0)
        run.seed_all(7)
        full = run.MetricEmT(tr.subjects, 7)
        run.train_stage(
            full, tr, va, cfg, run.teacher_objective, root / "uninterrupted", 7
        )
        run.seed_all(7)
        part = run.MetricEmT(tr.subjects, 7)
        run.DEADLINE = -1
        try:
            run.train_stage(
                part, tr, va, cfg, run.teacher_objective, root / "resumed", 7
            )
        except run.BudgetReached:
            pass
        else:
            raise AssertionError("Budget checkpoint was not exercised")
        run.DEADLINE = float("inf")
        fresh = run.MetricEmT(tr.subjects, 7)
        run.train_stage(fresh, tr, va, cfg, run.teacher_objective, root / "resumed", 7)
        a = torch.load(root / "uninterrupted/latest.pt", weights_only=False)
        b = torch.load(root / "resumed/latest.pt", weights_only=False)
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
        centers = torch.randn(4, 3, 15, 32)
        bank = run.Bank(centers, tr.subjects)
        z = torch.nn.functional.normalize(torch.randn(1, 32), dim=-1)
        meta = {"subject": [2], "trial": [1]}
        q = bank.target(z, meta)
        changed = centers.clone()
        changed[0] += 50
        changed[:, :, 0] -= 90
        torch.testing.assert_close(q, run.Bank(changed, tr.subjects).target(z, meta))
        torch.testing.assert_close(q.sum(1), torch.ones(1))
        print(
            "PASS: exact resume with optimizer/RNG restoration; reference exclusions; probability normalization."
        )


import unittest
import contextlib, io


class PipelineTests(unittest.TestCase):
    def test_incomplete_summary(self):
        from csrd_emt.__main__ import summarize

        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(Path(folder))
            summary = json.loads((Path(folder) / "summary.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertIsNone(summary["mean_ba"])

    def test_inference_reload_and_gradient_reversal(self):
        from csrd_emt.inference import export_student, load_student

        torch.set_num_threads(2)
        run.seed_all(12)
        model = run.ResearchEmT("erm").eval()
        x = torch.randn(2, 36, 62, 5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "student.pt"
            export_student(model, path)
            torch.testing.assert_close(model(x), load_student(path)(x), rtol=0, atol=0)
        value = torch.ones(3, requires_grad=True)
        run.Reverse.apply(value, 0.1).sum().backward()
        torch.testing.assert_close(value.grad, torch.full_like(value, -0.1))

    def test_synthetic_pipeline_and_resume(self):
        with contextlib.redirect_stdout(io.StringIO()):
            main()


if __name__ == "__main__":
    unittest.main()
