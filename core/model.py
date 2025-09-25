import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

def prepare_pipeline(model, tokenizer):
    """
    Prepare a HuggingFace text-generation pipeline with specified parameters.

    Args:
        model (BaseModelWithGenerate): A pretrained LLM with its hardware config (e.g. use quantization, floating point)
        tokenizer (AutoTokenizer): The tokenizer instance, must be compatible with model.

    Returns:
        HuggingFacePipeline: A prepared HuggingFace pipeline.
    """
    
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.2,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        return_full_text=False,
    )

def load_llm(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Load a HuggingFace language model and wrap it into a HuggingFacePipeline.

    Args:
        model_name (str, optional): The name of the model. See https://huggingface.co/models?pipeline_tag=text-generation&sort=trending for more. Defaults to "meta-llama/Llama-3.2-3B-Instruct".

    Returns:
        HuggingFacePipeLine: A wrapped LLM in HuggingFacePipeline.
    """

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return HuggingFacePipeline(pipeline=prepare_pipeline(model, tokenizer))
