"""SEED LOSO: fixed source-selected recipe; target opened only after model selection."""

from pathlib import Path
import copy, gc, hashlib, json, os, random, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from .data import (
    seed_all,
    MatNormalizer,
    FeatureDirectory,
    load_de,
    normalize_features,
    source_split,
    save,
    metrics,
    Data,
    evaluate,
)
from .models import (
    ChebyshevGraphConv,
    GraphEncoder,
    ModelConfig,
    FeedForward,
    TransformerBlock,
    TemporalAttention,
    AdaptedEmT,
    ResearchEmT,
    Reverse,
    MetricEmT,
    semantic_loss,
    Student,
)
from .reference import Bank, make_bank
from .objectives import teacher_objective, student_objective
from .inference import export_student

PROTOCOL = "csrd-seed-loso-v1-20261115"
BUDGET_SECONDS = 19800
DEADLINE = float("inf")
RUNTIME_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SOURCE_SHA = hashlib.sha256(
    b"".join(
        (Path(__file__).parent / n).read_bytes()
        for n in [
            "data.py",
            "models.py",
            "reference.py",
            "objectives.py",
            "inference.py",
        ]
    )
).hexdigest()
MAX_STEPS = 14000
VALIDATE_EVERY = 500
PATIENCE = 5
TEACHER_CONFIG = {"lr": 3e-4, "lambda_s": 0.1, "lambda_d": 0.0}
STUDENT_CONFIG = {"method": "csrd", "lr": 3e-4, "kd": 1.0, "adv": 0.1}
DEVICE = "cpu"


class BudgetReached(Exception):
    pass


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def identity(target):
    return dict(
        protocol=PROTOCOL,
        target=target,
        source_sha=SOURCE_SHA,
        runtime_sha=RUNTIME_SHA,
        teacher=TEACHER_CONFIG,
        student=STUDENT_CONFIG,
        max_steps=MAX_STEPS,
        validate_every=VALIDATE_EVERY,
        patience=PATIENCE,
        split=source_split(target),
        normalization="separate MinMax per subject/session; unlabeled target statistics permitted",
        refit=False,
        teacher_seed=2022 + target,
        student_seed=3022 + target,
    )


def train_stage(
    model,
    train_data,
    validation_data,
    config,
    objective_fn,
    output_dir,
    seed,
    teacher=None,
    bank=None,
    include_step_zero=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_identity = digest(
        dict(
            run=identity(TARGET_SUBJECT),
            config=config,
            seed=seed,
            include_step_zero=include_step_zero,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=0.01
    )
    rng = np.random.default_rng(seed)
    state = dict(
        step=0,
        best_key=None,
        best_state=None,
        best_score=None,
        best_prediction=None,
        best_step=None,
        stale=0,
        history=[],
    )
    latest = output_dir / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=DEVICE, weights_only=False)
        assert ck["identity"] == stage_identity, "Resume protocol mismatch"
        model.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        state = ck["state"]
        rng.bit_generator.state = ck["sampling_rng"]
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if DEVICE == "cuda":
            torch.cuda.set_rng_state_all([v.cpu() for v in ck["cuda_rng"]])

    def select(step):
        score, prediction = evaluate(model, validation_data, DEVICE)
        score["macro_f1"] = float(
            np.mean([v["macro_f1"] for v in score["subjects"].values()])
        )
        key = (score["ba"], -score["ce"], -step)
        if state["best_key"] is None or key > tuple(state["best_key"]):
            state.update(
                best_key=key,
                best_state={
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                best_score=score,
                best_prediction=prediction,
                best_step=step,
                stale=0,
            )
        else:
            state["stale"] += 1
        return score

    if state["step"] == 0 and state["best_state"] is None and include_step_zero:
        select(0)
    if not state.get("finished"):
        for step in range(state["step"] + 1, MAX_STEPS + 1):
            model.train()
            x, y, meta = train_data.batch(train_data.sample(rng), DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out = model(x, return_features=True)
            if teacher is None:
                loss, parts = objective_fn(model, out, y, meta, config, step)
            else:
                loss, parts = objective_fn(
                    model, out, y, meta, config, step, teacher, x, bank
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            if step % VALIDATE_EVERY == 0 or step == MAX_STEPS:
                score = select(step)
                state["step"] = step
                state["history"].append(
                    dict(
                        step=step,
                        loss=float(loss.detach()),
                        **parts,
                        validation_ba=score["ba"],
                        validation_f1=score["macro_f1"],
                    )
                )
                state["finished"] = step == MAX_STEPS or state["stale"] >= PATIENCE
                atomic_save(
                    latest,
                    dict(
                        identity=stage_identity,
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        state=state,
                        sampling_rng=rng.bit_generator.state,
                        python_rng=random.getstate(),
                        numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=(
                            torch.cuda.get_rng_state_all() if DEVICE == "cuda" else []
                        ),
                    ),
                )
                save(
                    output_dir / "progress.json",
                    dict(
                        step=step,
                        best_step=state["best_step"],
                        ba=score["ba"],
                        finished=state["finished"],
                    ),
                )
                if state["finished"]:
                    break
                if time.monotonic() > DEADLINE:
                    raise BudgetReached()
    assert state["best_state"] is not None
    model.load_state_dict(state["best_state"], strict=True)
    atomic_save(output_dir / "best.pt", state["best_state"])
    np.savez_compressed(
        output_dir / "validation_predictions.npz", **state["best_prediction"]
    )
    score = {**state["best_score"], "step": state["best_step"]}
    save(output_dir / "metrics.json", score)
    save(
        output_dir / "config.json",
        dict(
            **config,
            seed=seed,
            target=TARGET_SUBJECT,
            run_identity=identity(TARGET_SUBJECT),
        ),
    )
    history = pd.DataFrame(state["history"])
    history.to_csv(output_dir / "history.csv", index=False)
    return model, score, history


def verify_manifest(manifest, expected):
    for key, value in manifest.items():
        assert (
            value["feature_sha256"] == expected[key]["feature_sha256"]
        ), f"Data hash mismatch: {key}"


def verify_receipt(root):
    root = Path(root)
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["status"] == "COMPLETE"
    assert receipt["identity"] == digest(identity(receipt["target"]))
    for name, h in receipt["artifacts"].items():
        path = (root / name).resolve()
        assert path.is_relative_to(root.resolve())
        assert file_sha(path) == h, f"Artifact mismatch: {name}"
    return receipt


def run_fold(target, data_path, output, expected):
    global TARGET_SUBJECT
    TARGET_SUBJECT = target
    folder = Path(output) / f"fold_{target:02}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "receipt.json").exists():
        verify_receipt(folder)
        return json.loads((folder / "result.json").read_text())
    print(f"[FOLD_START] S{target:02}", flush=True)
    split = source_split(target)
    assert set(split["train"]).isdisjoint(split["val"])
    assert target not in split["train"] + split["val"]
    source, manifest = load_de(
        data_path, sessions=(1, 2, 3), excluded_subjects=[target]
    )
    assert len(manifest) == 42 and all(s != target for s, e in source)
    verify_manifest(manifest, expected)
    scaled, normalization = normalize_features({k: v["x"] for k, v in source.items()})
    records = {k: dict(v, x=scaled[k]) for k, v in source.items()}
    train_data = Data(records, split["train"])
    validation_data = Data(records, split["val"])
    run_identity = identity(target)
    if (folder / "run_identity.json").exists():
        assert json.loads((folder / "run_identity.json").read_text()) == run_identity
    save(folder / "run_identity.json", run_identity)
    save(
        folder / "source_manifest.json",
        dict(
            split=split,
            manifest=manifest,
            normalization=normalization,
            target_feature_reads=0,
        ),
    )
    teacher_seed = 2022 + target
    seed_all(teacher_seed)
    teacher = MetricEmT(train_data.subjects, teacher_seed).to(DEVICE)
    teacher, teacher_score, _ = train_stage(
        teacher,
        train_data,
        validation_data,
        TEACHER_CONFIG,
        teacher_objective,
        folder / "teacher",
        teacher_seed,
    )
    teacher.zero_grad(set_to_none=True)
    teacher.eval().requires_grad_(False)
    print(f'[TEACHER_SELECTED] S{target:02} step={teacher_score["step"]}', flush=True)
    bank, bank_metadata = make_bank(teacher, train_data, DEVICE)
    assert bank.subjects == split["train"]
    save(folder / "bank_metadata.json", bank_metadata)
    atomic_save(folder / "bank.pt", bank.centers)
    student_seed = 3022 + target
    seed_all(student_seed)
    student = Student(train_data.subjects, student_seed, teacher.state_dict()).to(
        DEVICE
    )
    student, student_score, _ = train_stage(
        student,
        train_data,
        validation_data,
        STUDENT_CONFIG,
        student_objective,
        folder / "student",
        student_seed,
        teacher=teacher,
        bank=bank,
        include_step_zero=True,
    )
    assert all(p.grad is None for p in teacher.parameters())
    # The target is first opened only after both checkpoints have been selected.
    student.eval()
    target_raw, target_manifest = load_de(
        data_path,
        sessions=(1, 2, 3),
        excluded_subjects=[s for s in range(1, 16) if s != target],
    )
    assert set(target_raw) == {(target, e) for e in [1, 2, 3]}
    assert not set(target_manifest) & set(manifest)
    verify_manifest(target_manifest, expected)
    target_scaled, target_normalization = normalize_features(
        {k: v["x"] for k, v in target_raw.items()}
    )
    test = Data(
        {k: dict(v, x=target_scaled[k]) for k, v in target_raw.items()}, [target]
    )
    inference = export_student(student, folder / "student_inference.pt")
    with torch.no_grad():
        x, _, _ = test.batch(range(min(4, len(test.rows))), DEVICE)
        torch.testing.assert_close(student(x), inference(x), rtol=1e-5, atol=1e-6)
    score, predictions = evaluate(inference, test, DEVICE)
    assert set(score["subjects"]) == {str(target)}
    assert len(set(predictions["sample_id"])) == len(predictions["sample_id"])
    assert all(v.startswith(f"S{target:02}_") for v in predictions["sample_id"])
    individual = score["subjects"][str(target)]
    np.savez_compressed(folder / "target_predictions.npz", **predictions)
    save(
        folder / "target_manifest.json",
        dict(
            manifest=target_manifest,
            normalization=target_normalization,
            access_phase="after source checkpoint selection",
            uses_target_labels_for_training=False,
        ),
    )
    result = dict(
        protocol=PROTOCOL,
        target=target,
        ba=individual["ba"],
        macro_f1=individual["macro_f1"],
        accuracy=individual["accuracy"],
        samples=individual["samples"],
        confusion=individual["confusion"],
        teacher_source_ba=teacher_score["ba"],
        student_source_ba=student_score["ba"],
        student_step=student_score["step"],
    )
    save(folder / "result.json", result)
    hashes = {
        str(p.relative_to(folder)): file_sha(p)
        for p in folder.rglob("*")
        if p.is_file() and p.name != "receipt.json"
    }
    save(
        folder / "receipt.json",
        dict(
            status="COMPLETE",
            target=target,
            identity=digest(run_identity),
            artifacts=hashes,
        ),
    )
    print("[TARGET_COMPLETE] " + json.dumps(result), flush=True)
    return result
