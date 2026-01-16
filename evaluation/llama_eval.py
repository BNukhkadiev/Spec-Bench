"""
Generate answers with local models (Llama 3.x Instruct ready)

- Uses HF tokenizer.apply_chat_template for proper Llama chat formatting
- Stops generation on <|eot_id|> (and eos) to avoid running to max_new_tokens
- Uses model.device (safe with device_map="auto")
"""

import json
import os
import time
import torch
import numpy as np
import shortuuid

from fastchat.llm_judge.common import load_questions
from tqdm import tqdm


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def _set_llama_stop_tokens(model, tokenizer):
    """
    Llama-3 Instruct commonly uses <|eot_id|> as end-of-turn.
    Stopping on it prevents the model from rambling until max_new_tokens.
    """
    _ensure_pad_token(tokenizer)

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # If token doesn't exist, don't hard-crash; just fall back to eos_token_id
    if isinstance(eot_id, int) and eot_id >= 0:
        model.generation_config.eos_token_id = [eot_id, tokenizer.eos_token_id]
    else:
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    model.generation_config.pad_token_id = tokenizer.pad_token_id


def build_inputs_llama(tokenizer, messages, model):
    """
    Build model inputs from chat messages using the tokenizer's chat template.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Tokenizer has no apply_chat_template(). "
            "Install a recent transformers/tokenizer for Llama-3 Instruct."
        )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt")
    return inputs.to(model.device)


def _decode_new_tokens(tokenizer, output_ids, prompt_len):
    gen_ids = output_ids[0][prompt_len:]
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
                questions[i: i + chunk_size],
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
    print("Check model training state:", model.training)
    print("CUDA VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # Set Llama stop tokens once
    _set_llama_stop_tokens(model, tokenizer)

    question0 = questions[0]

    # -------------------
    # Warmup
    # -------------------
    for _ in range(3):
        torch.manual_seed(0)

        # Fresh conversation
        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        for j in range(len(question0["turns"])):
            user_text = question0["turns"][j]
            messages.append({"role": "user", "content": user_text})

            inputs = build_inputs_llama(tokenizer, messages, model)
            prompt_len = inputs.input_ids.shape[-1]

            try:
                _sync_if_cuda()
                start_time = time.time()

                output_ids, new_token, step, accept_length_tree = forward_func(
                    inputs,
                    model,
                    tokenizer,
                    max_new_tokens,
                    **kwargs,
                )

                _sync_if_cuda()
                _ = time.time() - start_time

                output_text = _decode_new_tokens(tokenizer, output_ids, prompt_len)

            except RuntimeError as e:
                print("ERROR warmup question ID:", question0["question_id"], "|", repr(e))
                output_text = "ERROR"

            messages.append({"role": "assistant", "content": output_text})

    print("Warmup done")

    # -------------------
    # Main eval
    # -------------------
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

                inputs = build_inputs_llama(tokenizer, messages, model)
                prompt_len = inputs.input_ids.shape[-1]

                step = 0
                new_token = 0
                total_time = float("nan")
                accept_length_tree = []

                try:
                    _sync_if_cuda()
                    start_time = time.time()

                    output_ids, new_token, step, accept_length_tree = forward_func(
                        inputs,
                        model,
                        tokenizer,
                        max_new_tokens,
                        **kwargs,
                    )

                    _sync_if_cuda()
                    total_time = time.time() - start_time

                    output_text = _decode_new_tokens(tokenizer, output_ids, prompt_len)

                    if accept_length_tree:
                        cur_accept_lengths.extend(accept_length_tree)
                        accept_lengths_all.extend(accept_length_tree)

                except RuntimeError as e:
                    print("ERROR question ID:", question["question_id"], "|", repr(e))
                    output_text = "ERROR"
                    step = 0
                    new_token = 0
                    total_time = float("nan")
                    accept_length_tree = []

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
