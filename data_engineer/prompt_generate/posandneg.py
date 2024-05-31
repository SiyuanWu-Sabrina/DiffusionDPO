# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel


from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.models import CogAgentModel, CogVLMModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--version", type=str, default="chat", choices=['chat', 'vqa', 'chat_old', 'base'], help='version of language process. if there is \"text_processor_version\" in model_config.json, this option will be overwritten')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')

    parser.add_argument("--from_pretrained", type=str, default="/share/home/wusiyuan/imagereward_work/release/ImageReward-Aesthetic", 
                        help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="/share/home/wusiyuan/imagereward_work/prompt_generate/vicuna-7b-v1.5", 
                        help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--stream_chat", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--result_dir", type=str, default="/share/home/wusiyuan/imagereward_work/prompt_generate/result")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args = parser.parse_args()

    # load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cpu' if args.quant else 'cuda',
        **vars(args)
    ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else args.version
    print("[Language processor version]:", language_processor_version)
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=language_processor_version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
    
    if args.quant:
        quantize(model, args.quant)
        if torch.cuda.is_available():
            model = model.cuda()

    ### MAIN LOOP
    # model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    # load image path
    image_dir = "/share/home/wusiyuan/imagereward_work/prompt_generate/Stable-Diffusion-Prompts"
    image_dir = os.path.join(image_dir, f"generated_images_sdxl-{args.seed}")
    assert os.path.exists(image_dir), f"Image directory {image_dir} does not exist."
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]
    
    # result dict
    result_dict = {
        "image_path": [],
        "a": [],
        "b": [],
        "c": [],
    }

    # query
    queries = {
        "a": "Does the image have any physically implausible aspects? Only answer 'Yes' or 'No'.",
        "b": "Are there any objects in the image that seem odd? Only answer 'Yes' or 'No'.",
        "c": "Does this image make one feel overall displeasure or even psychological discomfort? Only answer 'Yes' or 'No'.",
    }

    # output redirection
    sys.stdout = open(os.path.join(args.result_dir, f"output_{args.seed}.txt"), "w")
    
    with torch.no_grad():
        # iterate through image path
        for image_path in image_paths:
            history = None
            cache_image = None
            
            image_path = [image_path] if rank == 0 else [None]
            if world_size > 1:
                torch.distributed.broadcast_object_list(image_path, 0)
            image_path = image_path[0]
            assert image_path is not None
            result_dict["image_path"].append(image_path)

            # change query
            for k, query in queries.items():
                query = [query] if rank == 0 else [None]
                if world_size > 1:
                    torch.distributed.broadcast_object_list(query, 0)
                query = query[0]
                assert query is not None
            
                try:
                    response, history, cache_image = chat(
                        image_path,
                        model,
                        text_processor_infer,
                        image_processor,
                        query,
                        history=history,
                        cross_img_processor=cross_image_processor,
                        image=cache_image,
                        max_length=args.max_length,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        invalid_slices=text_processor_infer.invalid_slices,
                        args=args
                        )
                except Exception as e:
                    print("Error: ", e)

                if rank == 0 and not args.stream_chat:
                    print(f"Model: [{k}] "+response)

                # process response
                result_dict[k].append(response)
            
            assert len(result_dict["image_path"]) == len(result_dict["a"]) == len(result_dict["b"]) == len(result_dict["c"])

    # dataframe store
    df = pd.DataFrame(result_dict)
    df.to_csv(os.path.join(args.result_dir, f"metrics_{args.seed}.csv"), index=False)


if __name__ == "__main__":
    main()