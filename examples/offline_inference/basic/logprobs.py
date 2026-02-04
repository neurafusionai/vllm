# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

from vllm import LLM, EngineArgs
from vllm.logprobs import Logprob
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        model="meta-llama/Llama-3.2-1B-Instruct",
        logprobs=5,
        prompt_logprobs=5,
    )
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    sampling_group.add_argument("--logprobs", type=int)
    sampling_group.add_argument("--prompt-logprobs", type=int)

    return parser


def _format_logprobs(logprobs_for_position: dict[int, Logprob] | None) -> str:
    if not logprobs_for_position:
        return "None"
    sorted_items = sorted(
        logprobs_for_position.items(),
        key=lambda item: item[1].logprob,
        reverse=True,
    )
    formatted = []
    for token_id, logprob in sorted_items[:3]:
        token_label = logprob.decoded_token or f"<id:{token_id}>"
        formatted.append(f"{token_label!r}:{logprob.logprob:.3f}")
    return ", ".join(formatted)


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    logprobs = args.pop("logprobs")
    prompt_logprobs = args.pop("prompt_logprobs")

    # Create an LLM
    llm = LLM(**args)

    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k
    if logprobs is not None:
        sampling_params.logprobs = logprobs
    if prompt_logprobs is not None:
        sampling_params.prompt_logprobs = prompt_logprobs

    prompts = [
        "Summarize the importance of unit tests in two sentences.",
        "Write a short haiku about latency.",
    ]
    outputs = llm.generate(prompts, sampling_params)

    print("-" * 80)
    for output in outputs:
        completion = output.outputs[0]
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated text: {completion.text!r}")

        if output.prompt_logprobs is not None:
            print("Prompt logprobs (first 3 positions):")
            for position, logprobs_for_position in itertools.islice(
                enumerate(output.prompt_logprobs), 3
            ):
                formatted = _format_logprobs(logprobs_for_position)
                print(f"  token {position}: {formatted}")

        if completion.logprobs is not None:
            print("Completion logprobs (first 3 positions):")
            for position, logprobs_for_position in itertools.islice(
                enumerate(completion.logprobs), 3
            ):
                formatted = _format_logprobs(logprobs_for_position)
                print(f"  token {position}: {formatted}")

        print("-" * 80)


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
