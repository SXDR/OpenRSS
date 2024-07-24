from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic
from segment_anything.utils import transforms


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q_1: nn.Module,
            linear_a_q_2: nn.Module,
            linear_a_q_3: nn.Module,
            linear_a_q_4: nn.Module,
            linear_b_q_1: nn.Module,
            linear_b_q_2: nn.Module,
            linear_b_q_3: nn.Module,
            linear_b_q_4: nn.Module,
            linear_a_v_1: nn.Module,
            linear_a_v_2: nn.Module,
            linear_a_v_3: nn.Module,
            linear_a_v_4: nn.Module,
            linear_b_v_1: nn.Module,
            linear_b_v_2: nn.Module,
            linear_b_v_3: nn.Module,
            linear_b_v_4: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q_1 = linear_a_q_1
        self.linear_a_q_2 = linear_a_q_2
        self.linear_a_q_3 = linear_a_q_3
        self.linear_a_q_4 = linear_a_q_4
        self.linear_b_q_1 = linear_b_q_1
        self.linear_b_q_2 = linear_b_q_2
        self.linear_b_q_3 = linear_b_q_3
        self.linear_b_q_4 = linear_b_q_4
        self.linear_a_v_1 = linear_a_v_1
        self.linear_a_v_2 = linear_a_v_2
        self.linear_a_v_3 = linear_a_v_3
        self.linear_a_v_4 = linear_a_v_4
        self.linear_b_v_1 = linear_b_v_1
        self.linear_b_v_2 = linear_b_v_2
        self.linear_b_v_3 = linear_b_v_3
        self.linear_b_v_4 = linear_b_v_4
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q_1 = self.linear_b_q_1(self.linear_a_q_1(x))
        new_q_2 = self.linear_b_q_2(self.linear_a_q_2(x))
        new_q_3 = self.linear_b_q_3(self.linear_a_q_3(x))
        new_q_4 = self.linear_b_q_4(self.linear_a_q_4(x))
        new_v_1 = self.linear_b_v_1(self.linear_a_v_1(x))
        new_v_2 = self.linear_b_v_2(self.linear_a_v_2(x))
        new_v_3 = self.linear_b_v_3(self.linear_a_v_3(x))
        new_v_4 = self.linear_b_v_4(self.linear_a_v_4(x))
        # new_q = (new_q_1 + new_q_2 + new_q_3 + new_q_4) / 4
        # new_v = (new_v_1 + new_v_2 + new_v_3 + new_v_4) / 4
        new_q = new_q_1 * 0.1 + new_q_2 * 0.2 + new_q_3 * 0.4 + new_q_4 * 0.3
        new_v = new_v_1 * 0.1 + new_v_2 * 0.2 + new_v_3 * 0.4 + new_v_4 * 0.3
        # new_q = new_q_1 * 0.1 + new_q_2 * 0.1 + new_q_3 * 0.4 + new_q_4 * 0.4
        # new_v = new_v_1 * 0.1 + new_v_2 * 0.1 + new_v_3 * 0.4 + new_v_4 * 0.4
        # new_q = new_q_1 * 0.1 + new_q_2 * 0.2 + new_q_3 * 0.5 + new_q_4 * 0.2
        # new_v = new_v_1 * 0.1 + new_v_2 * 0.2 + new_v_3 * 0.5 + new_v_4 * 0.2
        # new_q = new_q_1 * 0.1 + new_q_2 * 0.2 + new_q_3 * 0.6 + new_q_4 * 0.2
        # new_v = new_v_1 * 0.1 + new_v_2 * 0.2 + new_v_3 * 0.6 + new_v_4 * 0.2
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        # >>> model = ViT('B_16_imagenet1k')
        # >>> lora_model = LoRA_ViT(model, r=4)
        # >>> preds = lora_model(img)
        # >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
#        for param in sam_model.image_encoder.parameters():
 #           param.requires_grad = False
        for name, param in sam_model.image_encoder.named_parameters():
            if "patch_embed_prompt" in name or "prompt_blocks" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q_1 = nn.Linear(self.dim, 1, bias=False)
            w_a_linear_q_2 = nn.Linear(self.dim, 2, bias=False)
            w_a_linear_q_3 = nn.Linear(self.dim, 4, bias=False)
            w_a_linear_q_4 = nn.Linear(self.dim, 8, bias=False)
            w_b_linear_q_1 = nn.Linear(1, self.dim, bias=False)
            w_b_linear_q_2 = nn.Linear(2, self.dim, bias=False)
            w_b_linear_q_3 = nn.Linear(4, self.dim, bias=False)
            w_b_linear_q_4 = nn.Linear(8, self.dim, bias=False)
            w_a_linear_v_1 = nn.Linear(self.dim, 1, bias=False)
            w_a_linear_v_2 = nn.Linear(self.dim, 2, bias=False)
            w_a_linear_v_3 = nn.Linear(self.dim, 4, bias=False)
            w_a_linear_v_4 = nn.Linear(self.dim, 8, bias=False)
            w_b_linear_v_1 = nn.Linear(1, self.dim, bias=False)
            w_b_linear_v_2 = nn.Linear(2, self.dim, bias=False)
            w_b_linear_v_3 = nn.Linear(4, self.dim, bias=False)
            w_b_linear_v_4 = nn.Linear(8, self.dim, bias=False)
            self.w_As.append(w_a_linear_q_1)
            self.w_As.append(w_a_linear_q_2)
            self.w_As.append(w_a_linear_q_3)
            self.w_As.append(w_a_linear_q_4)
            self.w_Bs.append(w_b_linear_q_1)
            self.w_Bs.append(w_b_linear_q_2)
            self.w_Bs.append(w_b_linear_q_3)
            self.w_Bs.append(w_b_linear_q_4)
            self.w_As.append(w_a_linear_v_1)
            self.w_As.append(w_a_linear_v_2)
            self.w_As.append(w_a_linear_v_3)
            self.w_As.append(w_a_linear_v_4)
            self.w_Bs.append(w_b_linear_v_1)
            self.w_Bs.append(w_b_linear_v_2)
            self.w_Bs.append(w_b_linear_v_3)
            self.w_Bs.append(w_b_linear_v_4)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q_1,
                w_a_linear_q_2,
                w_a_linear_q_3,
                w_a_linear_q_4,
                w_b_linear_q_1,
                w_b_linear_q_2,
                w_b_linear_q_3,
                w_b_linear_q_4,
                w_a_linear_v_1,
                w_a_linear_v_2,
                w_a_linear_v_3,
                w_a_linear_v_4,
                w_b_linear_v_1,
                w_b_linear_v_2,
                w_b_linear_v_3,
                w_b_linear_v_4
            )
        self.reset_parameters()
        self.sam = sam_model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}
        patch_embed_prompt = {}
        prompt_blocks = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam,
                                                                     torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value
            if 'patch_embed_prompt' in key:
                patch_embed_prompt[key] = value
            if 'prompt_blocks' in key:
                prompt_blocks[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors, **patch_embed_prompt, **prompt_blocks}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        
        # load patch_embed_prompt
        patch_embed_prompt_keys = [k for k in sam_keys if 'patch_embed_prompt' in k]
        patch_embed_prompt_values = [state_dict[k] for k in patch_embed_prompt_keys]
        patch_embed_prompt_new_state_dict = {k: v for k, v in zip(patch_embed_prompt_keys, patch_embed_prompt_values)}
        sam_dict.update(patch_embed_prompt_new_state_dict)

        # load prompt_blocks
        prompt_blocks_keys = [k for k in sam_keys if 'prompt_blocks' in k]
        prompt_blocks_values = [state_dict[k] for k in prompt_blocks_keys]
        prompt_blocks_new_state_dict = {k: v for k, v in zip(prompt_blocks_keys, prompt_blocks_values)}
        sam_dict.update(prompt_blocks_new_state_dict)
        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    # def forward(self, batched_input, multimask_output, image_size):
    #     return self.sam(batched_input, multimask_output, image_size)
    def forward(self, batched_input, points, clip_text_feature, multimask_output):
        return self.sam(batched_input, points, clip_text_feature, multimask_output)

    # def forward(self, x: Tensor) -> Tensor:
    #     return self.lora_vit(x)


