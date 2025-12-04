
import warnings

import nncore
import torch
import torch.nn as nn
from peft import PeftModel
from safetensors.torch import load_model
from transformers import AutoConfig, AutoModel, AutoProcessor, GenerationConfig, Qwen2VLForConditionalGeneration


def get_auto_device():
    try:
        import torch_npu
        has_npu = torch_npu.npu.is_available()
    except ImportError:
        has_npu = False

    return 'cuda' if torch.cuda.is_available() else 'npu' if has_npu else 'cpu'


def build_model(model_path, config=None, is_trainable=False, merge_adapter=False, device='auto', dtype=torch.float16):
    # set do_resize to false to avoid duplicated resizing
    # https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
    processor = AutoProcessor.from_pretrained(model_path, do_resize=False)
    """
    processor就是将输入文本和视频或图像转换成token并拼接在一起的模块,
    还没有送入到transformer中
    """

    # eager attention has known & unknown bugs
    # [4.46.2] broken causality fp16: https://github.com/huggingface/transformers/issues/35151
    # [4.48.1] broken sliding window: https://github.com/huggingface/transformers/issues/35924
    attn_implementation = 'sdpa'
    """
    attn_implementation规定了transformer注意力实现方式，除此之外还有很多实现方式
    比如eager，flash_attention_2，他们其实是实现同一个数学公式的不同策略，区别主要
    体现在速度，显存占用，兼容性上
    """

    config = config or AutoConfig.from_pretrained(model_path)
    """
    config是模型配置，存放了模型的超参数和结构信息，比如模型类型：model_type="qwen2_vl"
    隐藏层维度：hidden_size=4096等
    """

    adapter_path = nncore.join(model_path, getattr(config, 'role', 'unknown'))
    partial_path = nncore.join(model_path, 'pytorch_model.safetensors')
    """
    adapter_path是LoRA适配器的路径
    partial_path是模型权重文件
    """

    """
    这段代码是在加载基础模型，还没有挂上LoRA和权重文件，并且这个模型也只是建立骨架，部分真正的权重参数还没有加载进来
    """
    if nncore.is_dir(adapter_path) or nncore.is_file(partial_path):
        print(f'Loading base model from {config.base_model_path}...')
        model = AutoModel.from_pretrained(
            config.base_model_path,
            config=config,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
            device_map='auto' if device == 'all' else None)

        try:
            model.generation_config = GenerationConfig.from_pretrained(model_path)
        except OSError:
            warnings.warn('generation_config.json not found')

        """
        这段代码是把meta参数“实体化”为空的张量（meta参数是之前from_pretrained加载时因为low_cpu_mem_usage等问题而没有加载进来的参数），
        方便后续操作，还没有把真实的数值权重加载进来
        """
        meta_state_dict = {
            n: torch.empty_like(p, device='cpu')
            for n, p in model.named_parameters() if p.device == torch.device('meta')
        }
        model.load_state_dict(meta_state_dict, strict=False, assign=True)

        """
        下面这两段size的代码都是在对齐形状，为后面铺路
        """
        size = (model.model.embed_tokens.num_embeddings, model.model.embed_tokens.embedding_dim)
        if model.model.embed_tokens.weight.size() != size:
            print(f'Resizing embed_tokens to {size}...')
            model.model.embed_tokens.weight = nn.Parameter(model.model.embed_tokens.weight.new_empty(size))

        size = (model.lm_head.out_features, model.lm_head.in_features)
        if model.lm_head.weight.size() != size:
            print(f'Resizing lm_head to {size}...')
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.new_empty(size))

        """
        这段代码就是把LoRA适配器挂到已加载好的基座模型上
        """
        if nncore.is_dir(adapter_path):
            print(f'Loading adapter from {adapter_path}...')
            # transformers integration does not support merge_and_unload, use peft instead
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                adapter_name=config.role,
                is_trainable=is_trainable,
                low_cpu_mem_usage=True,
                # load adapters to the same device as embed_tokens
                torch_device=str(model.model.embed_tokens.weight.device))

        """
        把safetensors里的参数加载进当前模型
        """
        if nncore.is_file(partial_path):
            print(f'Loading state dict from {partial_path}...')
            _, unexpected = load_model(model, partial_path, strict=False, device=str(model.device))
            assert len(unexpected) == 0, f'unexpected parameters: {unexpected}'

        """
        这段代码就是把已经挂在基座上的LoRA增量合并进基座模型真实参数中
        """
        if merge_adapter and nncore.is_dir(adapter_path):
            print('Merging adapter and unloading...')
            model = model.merge_and_unload()
            model._hf_peft_config_loaded = False
    else:
        print(f'Loading full model from {model_path}...')

        "如果没有LoRA，也没有单独的partial权重文件，那就加载完整的模型"
        if len(config.architectures) == 1 and config.model_type == 'qwen2_vl':
            model_cls = Qwen2VLForConditionalGeneration
        else:
            model_cls = AutoModel

        model = model_cls.from_pretrained(
            model_path,
            config=config,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
            device_map='auto' if device == 'all' else None)

    if not is_trainable and device != 'all':
        device = get_auto_device() if device == 'auto' else device
        model = model.to(device).eval()

    return model, processor
