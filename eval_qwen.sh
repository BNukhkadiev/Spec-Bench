Qwen_PATH=Qwen/Qwen2.5-7B-Instruct

Drafter_PATH=Qwen/Qwen2.5-1.5B-Instruct
# Space_PATH=/your_own_path/vicuna-v1.3-7b-space
datastore_PATH=./model/rest/datastore/datastore_chat_large.idx

MODEL_NAME=Qwen2.5-7B-Instruct

TEMP=0.0
GPU_DEVICES=0

bench_NAME="sentiment_analysis"
# bench_NAME="multiclass_product_classification"

torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.qwen_inference_sps --model-path $Qwen_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-68m-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
