"""
Generate answers with local models (Qwen2.5-ready).

Replaces FastChat Vicuna prompting with Hugging Face chat templates:
- tokenizer.apply_chat_template(..., add_generation_prompt=True)

Keeps the same output JSONL format as the original script.
"""

import json
import os
import time
import torch
import numpy as np
import shortuuid

from fastchat.llm_judge.common import load_questions
from tqdm import tqdm


def _ensure_pad_token(tokenizer):
    # Avoid padding/generation edge cases
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_inputs(tokenizer, messages, device):
    """
    Build model inputs from chat messages using the tokenizer's chat template.
    Qwen2.5 Instruct supports apply_chat_template.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # important: tells model to generate assistant reply
        )
    else:
        # Fallback (not ideal, but prevents hard crash if someone uses a non-chat tokenizer)
        # For Qwen2.5 you SHOULD have apply_chat_template.
        text = ""
        for m in messages:
            text += f"{m['role']}: {m['content']}\n"
        text += "assistant:"

    inputs = tokenizer([text], return_tensors="pt")
    return inputs.to(device)


def _decode_generated(tokenizer, output_ids, input_len):
    gen_ids = output_ids[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def run_eval(
    model,
    tokenizer,
    forward_func,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_tokens,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    **kwargs,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(get_model_answers).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                tokenizer,
                forward_func,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_tokens,
                num_choices,
                **kwargs,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model,
    tokenizer,
    forward_func,
    model_id,
    questions,
    answer_file,
    max_new_tokens,
    num_choices,
    **kwargs,
):
    model.eval()
    _ensure_pad_token(tokenizer)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Check model training state:", model.training)
    print("CUDA VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # With device_map="auto", inputs should go to model.device (first device)
    device = model.device

    question0 = questions[0]

    # --------------------
    # Warmup
    # --------------------
    for _ in range(3):
        torch.manual_seed(0)

        # Start a fresh chat for warmup
        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        for j in range(len(question0["turns"])):
            user_text = question0["turns"][j]
            messages.append({"role": "user", "content": user_text})

            inputs = _build_inputs(tokenizer, messages, device)
            input_len = inputs.input_ids.shape[-1]

            try:
                _sync_if_cuda()
                start_time = time.time()

                output_ids, new_token, step, _accept_len = forward_func(
                    inputs, model, tokenizer, max_new_tokens, **kwargs
                )

                _sync_if_cuda()
                _ = time.time() - start_time

                output_text = _decode_generated(tokenizer, output_ids, input_len)
            except RuntimeError as e:
                print("ERROR warmup question ID:", question0["question_id"], "|", repr(e))
                output_text = "ERROR"

            messages.append({"role": "assistant", "content": output_text})

    print("Warmup done")

    # --------------------
    # Main eval
    # --------------------
    accept_lengths_all = []

    for question in tqdm(questions):
        choices = []

        for i in range(num_choices):
            torch.manual_seed(i)

            messages = [{"role": "system", "content": "You are a helpful assistant."}]

            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            cur_accept_lengths = []

            for j in range(len(question["turns"])):
                user_text = question["turns"][j]
                messages.append({"role": "user", "content": user_text})

                inputs = _build_inputs(tokenizer, messages, device)
                input_len = inputs.input_ids.shape[-1]

                step = 0
                new_token = 0
                total_time = float("nan")
                accept_length_list = []

                try:
                    _sync_if_cuda()
                    start_time = time.time()

                    output_ids, new_token, step, accept_length_list = forward_func(
                        inputs, model, tokenizer, max_new_tokens, **kwargs
                    )

                    _sync_if_cuda()
                    total_time = time.time() - start_time

                    output_text = _decode_generated(tokenizer, output_ids, input_len)

                    # Collect speculative stats if provided
                    if accept_length_list:
                        cur_accept_lengths.extend(accept_length_list)
                        accept_lengths_all.extend(accept_length_list)

                except RuntimeError as e:
                    print("ERROR question ID:", question["question_id"], "|", repr(e))
                    output_text = "ERROR"
                    step = 0
                    new_token = 0
                    total_time = float("nan")
                    accept_length_list = []

                turns.append(output_text)
                steps.append(int(step))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)

                messages.append({"role": "assistant", "content": output_text})

            choices.append(
                {
                    "index": i,
                    "turns": turns,
                    "decoding_steps": steps,
                    "new_tokens": new_tokens,
                    "wall_time": wall_time,
                    "accept_lengths": cur_accept_lengths,
                }
            )

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

    if len(accept_lengths_all) > 0:
        print("#Mean accepted tokens:", float(np.mean(accept_lengths_all)))
    else:
        print("#Mean accepted tokens: n/a (no accept_lengths returned)")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])
