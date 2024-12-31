# Licensed under the MIT license.

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
import math


def load_vLLM_model(model_ckpt, seed, tensor_parallel_size=1, half_precision=False, max_num_seqs=128):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    if half_precision:
        llm = LLM(
            model=model_ckpt,
            dtype="half",
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
            enforce_eager=True
        )
    else:
        llm = LLM(
            model=model_ckpt,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
            enforce_eager=True

        )

    return tokenizer, llm


def generate_with_vLLM_model(
    model,
    input,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    n=1,
    max_tokens=256,
    logprobs=1,
    stop=[],
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        n=n,
        logprobs=logprobs,
        max_tokens=max_tokens,
        stop=stop,
    )

    output = model.generate(input, sampling_params, use_tqdm=False)
    return output

#debugging code
if __name__ == "__main__":
    model_ckpt = "meta-llama/Llama-3.1-8B"
    tokenizer, model = load_vLLM_model(model_ckpt, seed=42, tensor_parallel_size=4, half_precision=False,max_num_seqs=256)
    input = "What is the meaning of life?"
    output = generate_with_vLLM_model(model, input)
    #for debugging only
    #breakpoint()
    print(output[0].outputs[0].text)
