import argparse
import json
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------- IO ----------------
def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_refs(refs_path):
    # refs: {"question_id": ..., "task": ..., "reference_output": ...}
    refs = {}
    task_by_qid = {}
    for r in read_jsonl(refs_path):
        qid = int(r["question_id"])
        refs[qid] = r.get("reference_output")
        task_by_qid[qid] = r.get("task") or r.get("category") or ""
    return refs, task_by_qid


def load_answers(answers_path):
    # FastChat answers: {"question_id":..., "choices":[{"turns":[...]}], "model_id":...}
    answers = {}
    model_id = None
    for a in read_jsonl(answers_path):
        qid = int(a["question_id"])
        choices = a.get("choices") or []
        if not choices:
            answers[qid] = ""
        else:
            turns = choices[0].get("turns") or []
            answers[qid] = turns[-1] if turns else ""
        if model_id is None:
            model_id = a.get("model_id")
    return answers, model_id


# ---------------- Normalization ----------------
def normalize_task_name(task: str) -> str:
    t = (task or "").strip()
    if t == "Query_Product_Rank":
        return "Query_Product_Ranking"
    return t


# ---------------- Parsing ----------------
LABEL_LETTERS = set("ABCDE")

def extract_first_option_letter(text):
    """
    Extract A-E from model answer. Accepts:
    "A", "A:", "(A)", "Answer: A", "I choose A", etc.
    """
    if text is None:
        return None
    t = str(text).strip()

    # Find standalone A-E anywhere
    m = re.search(r"\b([A-E])\b", t)
    if m:
        return m.group(1)

    # Or at start like "A:" / "(A)"
    m = re.match(r"^\s*[\(\[]?\s*([A-E])\s*[:\)\]]?", t)
    if m:
        return m.group(1)

    return None


def parse_ref_label_letter(ref_output):
    """
    Reference output often like: "A: ...."
    We take the first A-E we find.
    """
    if ref_output is None:
        return None
    return extract_first_option_letter(str(ref_output)) or (str(ref_output).strip()[:1] if str(ref_output).strip() else None)


# ---------------- Metrics ----------------
def eval_multiclass(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def DCG(score_list):
    dcg = 0.0
    for i, s in enumerate(score_list):
        dcg += (2 ** s - 1) / (np.log2(i + 2))
    return dcg


def eval_ndcg(pred_text, ref_output):
    """
    Ranking evaluation:
    - Prediction: "A,B,C,D" (ranking letters)
    - Reference output: must allow mapping option->label among {E,S,C,I}
      Common forms:
        * JSON list e.g. ["E","S","C","I"] aligns to A,B,C,D
        * JSON dict e.g. {"A":"E","B":"S","C":"C","D":"I"}
    We map E/S/C/I -> numeric and compute NDCG from predicted order.
    """
    score_mapping = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

    if ref_output is None:
        return None, 1

    # Parse reference structure
    label2score = {}
    parsed = None

    # ref_output may be a JSON string
    if isinstance(ref_output, str):
        s = ref_output.strip()
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = None
    else:
        parsed = ref_output

    if isinstance(parsed, list):
        opt = "A"
        for lab in parsed:
            lab = str(lab).strip()
            label2score[opt] = score_mapping.get(lab, None)
            opt = chr(ord(opt) + 1)

    elif isinstance(parsed, dict):
        for k, v in parsed.items():
            kk = str(k).strip()
            vv = str(v).strip()
            if kk and kk[0].isalpha():
                letter = kk[0].upper()
                label2score[letter] = score_mapping.get(vv, None)
    else:
        # Can't evaluate if reference doesn't define per-option relevance labels
        return None, 1

    # Parse prediction ranking
    ranks = [x.strip() for x in str(pred_text).strip().split(",") if x.strip()]
    if not ranks:
        return None, 1

    invalid = 0
    scores = []
    for r in ranks:
        letter = extract_first_option_letter(r) or (r[0].upper() if r else None)
        if not letter or letter not in label2score or label2score[letter] is None:
            invalid += 1
            continue
        scores.append(label2score[letter])

    if not scores:
        return None, 1

    ideal_scores = sorted(scores, reverse=True)
    dcg = DCG(scores)
    idcg = DCG(ideal_scores)
    return float(dcg / (idcg + 1e-9)), invalid


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="FastChat answers JSONL")
    ap.add_argument("--refs", required=True, help="Refs JSONL (e.g., Sentiment_Analysis_refs.jsonl or merged)")
    ap.add_argument("--task", default="", help="Optional: restrict to one of: Sentiment_Analysis, Multiclass_Product_Classification, Query_Product_Rank")
    args = ap.parse_args()

    refs, task_by_qid = load_refs(args.refs)
    answers, model_id = load_answers(args.answers)

    qids = sorted(set(refs.keys()) & set(answers.keys()))
    if not qids:
        raise SystemExit("No overlapping question_ids between answers and refs.")

    # Group by task
    groups = defaultdict(list)
    for qid in qids:
        groups[normalize_task_name(task_by_qid.get(qid, ""))].append(qid)

    # Optional filter
    if args.task:
        wanted = normalize_task_name(args.task)
        groups = {k: v for k, v in groups.items() if normalize_task_name(k) == wanted}
        if not groups:
            raise SystemExit(f"No questions found for task={args.task} after filtering.")

    results = {"model_id": model_id, "by_task": {}}

    for task, tqids in groups.items():
        task = normalize_task_name(task)

        if task in {"Sentiment_Analysis", "Multiclass_Product_Classification"}:
            y_true, y_pred = [], []
            invalid = 0
            for qid in tqids:
                ref_letter = parse_ref_label_letter(refs[qid])
                pred_letter = extract_first_option_letter(answers[qid])

                if (ref_letter not in LABEL_LETTERS) or (pred_letter not in LABEL_LETTERS):
                    invalid += 1
                    continue
                y_true.append(ref_letter)
                y_pred.append(pred_letter)

            metrics = eval_multiclass(y_true, y_pred) if y_true else {}
            results["by_task"][task] = {
                **metrics,
                "n_total": len(tqids),
                "n_used": len(y_true),
                "n_invalid": invalid,
            }

        elif task == "Query_Product_Ranking":
            ndcgs = []
            invalid = 0
            invalid_ranks = 0

            for qid in tqids:
                ndcg_val, inv_i = eval_ndcg(answers[qid], refs[qid])
                if ndcg_val is None:
                    invalid += 1
                    continue
                ndcgs.append(ndcg_val)
                invalid_ranks += inv_i

            results["by_task"][task] = {
                "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
                "n_total": len(tqids),
                "n_used": len(ndcgs),
                "n_invalid": invalid,
                "invalid_ranks": int(invalid_ranks),
            }

        else:
            # Ignore any other tasks
            continue

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
