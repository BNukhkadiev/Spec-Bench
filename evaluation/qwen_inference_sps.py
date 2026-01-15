"""
Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path Qwen/Qwen2.5-7B-Instruct --drafter-path Qwen/Qwen2.5-1.5B-Instruct --model-id qwen2.5-7b_sps
"""
import argparse

from evaluation.eval import run_eval, reorg_answer_file
from fastchat.utils import str_to_torch_dtype

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin
from model.sps.decoding import assisted_decoding


def _ensure_pad_token(tokenizer):
    # Avoid generation/padding edge cases
    if tokenizer.pad_token_id is None:
        # Common practice: use eos as pad for decoder-only LMs
        tokenizer.pad_token = tokenizer.eos_token


def _check_tokenizer_compat(tokenizer, assistant_tokenizer):
    """
    Best case: identical tokenizers (same vocab + same special tokens).
    If not identical, we can still run using Universal Assisted Decoding (UAD)
    by passing both tokenizers to generate().
    """
    same_vocab = (len(tokenizer) == len(assistant_tokenizer))
    same_specials = (
        tokenizer.eos_token_id == assistant_tokenizer.eos_token_id
        and tokenizer.bos_token_id == assistant_tokenizer.bos_token_id
        and tokenizer.pad_token_id == assistant_tokenizer.pad_token_id
    )
    return same_vocab and same_specials


def sps_forward(
    inputs,
    model,
    tokenizer,
    max_new_tokens,
    do_sample=False,
    temperature=0.0,
    drafter=None,
    drafter_tokenizer=None,  # keep parameter so your call signature stays compatible
):
    print("Target vocab:", len(tokenizer), "Drafter vocab:", len(drafter_tokenizer))
    input_ids = inputs.input_ids
    model.generation_config.max_new_tokens = max_new_tokens

    output_ids, idx, accept_length_list = model.generate(
        **inputs,
        generation_config=model.generation_config,
        assistant_model=drafter,
        do_sample=do_sample,
        temperature=temperature,
    )

    new_token = len(output_ids[0][len(input_ids[0]):])
    return output_ids, new_token, idx + 1, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--drafter-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
    )
    args = parser.parse_args()

    # Hook in your custom assisted decoding
    GenerationMixin.assisted_decoding = assisted_decoding

    question_file = f"data/{args.bench_name}/question.jsonl"
    answer_file = args.answer_file or f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"
    print(f"Output to {answer_file}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    drafter = AutoModelForCausalLM.from_pretrained(
        args.drafter_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    # Load both tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    drafter_tokenizer = AutoTokenizer.from_pretrained(args.drafter_path)

    _ensure_pad_token(tokenizer)
    _ensure_pad_token(drafter_tokenizer)

    # Informative check (don’t hard-fail; UAD can handle mismatch if you pass both tokenizers)
    identical = _check_tokenizer_compat(tokenizer, drafter_tokenizer)
    print(f"Tokenizer identical (vocab+special tokens): {identical}")

    model.eval()
    drafter.eval()

    do_sample = args.temperature > 0

    run_eval(
        model=model,
        tokenizer=tokenizer,                 # target tokenizer (used to build inputs/prompt)
        forward_func=sps_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        drafter=drafter,
        drafter_tokenizer=drafter_tokenizer,  # <-- pass into sps_forward
        temperature=args.temperature,
        do_sample=do_sample,
    )

    reorg_answer_file(answer_file)
