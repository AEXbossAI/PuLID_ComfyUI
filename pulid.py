import torch
from torch import nn
import torchvision.transforms as T
import torch.nn.functional as F
import os
import math
import numpy as np
import folder_paths
import comfy.utils
import comfy.model_management
from insightface.app import FaceAnalysis
try:
    from facexlib.parsing import init_parsing_model
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    FACEXLIB_AVAILABLE = True
except ImportError:
    FACEXLIB_AVAILABLE = False
from comfy.ldm.modules.attention import optimized_attention

from .eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from .encoders import IDEncoder

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

MODELS_DIR = os.path.join(folder_paths.models_dir, "pulid")
os.makedirs(MODELS_DIR, exist_ok=True)
# Only register .safetensors — prevents ComfyUI from scanning .pt files (EVA-CLIP OOM)
# Also search loras/ so the model can be placed there during before_start (pulid/ doesn't exist yet then)
_loras_dirs = folder_paths.folder_names_and_paths.get("loras", ([],))[0]
folder_paths.folder_names_and_paths["pulid"] = ([MODELS_DIR] + list(_loras_dirs), {'.safetensors'})


class PulidModel(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.image_proj_model = self.init_id_adapter()
        self.image_proj_model.load_state_dict(model["image_proj"])
        self.ip_layers = To_KV(model["ip_adapter"])

    def init_id_adapter(self):
        image_proj_model = IDEncoder()
        return image_proj_model

    def get_image_embeds(self, face_embed, clip_embeds):
        embeds = self.image_proj_model(face_embed, clip_embeds)
        return embeds

class To_KV(nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[key.replace(".weight", "").replace(".", "_")].weight.data = value

def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image

def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor

def tensor_to_size(source, dest_size):
    if isinstance(dest_size, torch.Tensor):
        dest_size = dest_size.shape[0]
    source_size = source.shape[0]

    if source_size < dest_size:
        shape = [dest_size - source_size] + [1]*(source.dim()-1)
        source = torch.cat((source, source[-1:].repeat(shape)), dim=0)
    elif source_size > dest_size:
        source = source[:dest_size]

    return source

def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    else:
        to["patches_replace"]["attn2"] = to["patches_replace"]["attn2"].copy()

    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = Attn2Replace(pulid_attention, **patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["attn2"][key].add(pulid_attention, **patch_kwargs)

class Attn2Replace:
    def __init__(self, callback=None, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]

    def add(self, callback, **kwargs):
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, q, k, v, extra_options):
        dtype = q.dtype
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9

        for i, callback in enumerate(self.callback):
            if sigma <= self.kwargs[i]["sigma_start"] and sigma >= self.kwargs[i]["sigma_end"]:
                out = out + callback(out, q, k, v, extra_options, **self.kwargs[i])

        return out.to(dtype=dtype)

def pulid_attention(out, q, k, v, extra_options, module_key='', pulid=None, cond=None, uncond=None, weight=1.0, ortho=False, ortho_v2=False, mask=None, **kwargs):
    k_key = module_key + "_to_k_ip"
    v_key = module_key + "_to_v_ip"

    dtype = q.dtype
    seq_len = q.shape[1]
    cond_or_uncond = extra_options["cond_or_uncond"]
    b = q.shape[0]
    batch_prompt = b // len(cond_or_uncond)
    _, _, oh, ow = extra_options["original_shape"]

    k_cond = pulid.ip_layers.to_kvs[k_key](cond).repeat(batch_prompt, 1, 1)
    k_uncond = pulid.ip_layers.to_kvs[k_key](uncond).repeat(batch_prompt, 1, 1)
    v_cond = pulid.ip_layers.to_kvs[v_key](cond).repeat(batch_prompt, 1, 1)
    v_uncond = pulid.ip_layers.to_kvs[v_key](uncond).repeat(batch_prompt, 1, 1)
    ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
    ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)

    out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])

    if ortho:
        out = out.to(dtype=torch.float32)
        out_ip = out_ip.to(dtype=torch.float32)
        projection = (torch.sum((out * out_ip), dim=-2, keepdim=True) / torch.sum((out * out), dim=-2, keepdim=True) * out)
        orthogonal = out_ip - projection
        out_ip = weight * orthogonal
    elif ortho_v2:
        out = out.to(dtype=torch.float32)
        out_ip = out_ip.to(dtype=torch.float32)
        attn_map = q @ ip_k.transpose(-2, -1)
        attn_mean = attn_map.softmax(dim=-1).mean(dim=1, keepdim=True)
        attn_mean = attn_mean[:, :, :5].sum(dim=-1, keepdim=True)
        projection = (torch.sum((out * out_ip), dim=-2, keepdim=True) / torch.sum((out * out), dim=-2, keepdim=True) * out)
        orthogonal = out_ip + (attn_mean - 1) * projection
        out_ip = weight * orthogonal
    else:
        out_ip = out_ip * weight

    if mask is not None:
        mask_h = oh / math.sqrt(oh * ow / seq_len)
        mask_h = int(mask_h) + int((seq_len % int(mask_h)) != 0)
        mask_w = seq_len // mask_h

        mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
        mask = tensor_to_size(mask, batch_prompt)

        mask = mask.repeat(len(cond_or_uncond), 1, 1)
        mask = mask.view(mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])

        mask_len = mask_h * mask_w
        if mask_len < seq_len:
            pad_len = seq_len - mask_len
            pad1 = pad_len // 2
            pad2 = pad_len - pad1
            mask = F.pad(mask, (0, 0, pad1, pad2), value=0.0)
        elif mask_len > seq_len:
            crop_start = (mask_len - seq_len) // 2
            mask = mask[:, crop_start:crop_start+seq_len, :]

        out_ip = out_ip * mask

    return out_ip.to(dtype=dtype)

def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x

# ArcFace 5-point reference landmarks (standard 112x112 space)
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

def _align_face_no_facexlib(image_np, face_obj, size=512):
    """Align face via InsightFace keypoints — fallback when facexlib unavailable."""
    import cv2
    kps = face_obj.kps.astype(np.float32)
    scale = size / 112.0
    dst = _ARCFACE_DST * scale
    M, _ = cv2.estimateAffinePartial2D(kps, dst, method=cv2.LMEDS)
    if M is None:
        # Fallback: plain bbox crop
        x1, y1, x2, y2 = map(int, face_obj.bbox)
        crop = image_np[max(0,y1):y2, max(0,x1):x2]
        return cv2.resize(crop, (size, size))
    # image_np is BGR (from tensor_to_image), warpAffine works directly
    aligned = cv2.warpAffine(image_np, M, (size, size), borderValue=(255, 255, 255))
    return aligned


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

class PulidModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pulid_file": (folder_paths.get_filename_list("pulid"), )}}

    RETURN_TYPES = ("PULID",)
    FUNCTION = "load_model"
    CATEGORY = "pulid"

    def load_model(self, pulid_file):
        ckpt_path = folder_paths.get_full_path("pulid", pulid_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model

        model = PulidModel(model)

        return (model,)

class PulidInsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM", "CoreML"], ),
            },
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_insightface"
    CATEGORY = "pulid"

    def load_insightface(self, provider):
        model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',])
        model.prepare(ctx_id=0, det_size=(640, 640))

        return (model,)

class PulidEvaClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        clip_files = folder_paths.get_filename_list("clip_vision")
        eva_files = [f for f in clip_files if "EVA" in f or "eva" in f.lower()]
        # Also list EVA files placed directly in /tmp/
        for tmp_dir in ["/tmp", "/tmp/eva_clip"]:
            if os.path.isdir(tmp_dir):
                for f in os.listdir(tmp_dir):
                    if ("EVA" in f or "eva" in f.lower()) and f not in eva_files:
                        eva_files.append(f)
        return {"required": {
            "model": (["auto"] + eva_files,),
        }}

    RETURN_TYPES = ("EVA_CLIP",)
    FUNCTION = "load_eva_clip"
    CATEGORY = "pulid"

    def load_eva_clip(self, model="auto"):
        from .eva_clip.factory import create_model_and_transforms

        pretrained = "eva_clip"
        # Search order: /tmp/ → /tmp/eva_clip/ → models/pulid/ → clip_vision/
        search_dirs = [
            "/tmp",
            "/tmp/eva_clip",
            MODELS_DIR,
            "/opt/ComfyUI/tmp/eva_clip",
        ] + folder_paths.get_folder_paths("clip_vision")

        if model == "auto":
            for d in search_dirs:
                if os.path.isdir(d):
                    for f in sorted(os.listdir(d)):
                        if "EVA" in f or "eva" in f.lower():
                            pretrained = os.path.join(d, f)
                            break
                if pretrained != "eva_clip":
                    break
        else:
            for d in search_dirs:
                candidate = os.path.join(d, model)
                if os.path.exists(candidate):
                    pretrained = candidate
                    break

        model_obj, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', pretrained, force_custom_clip=True)
        model_obj = model_obj.visual

        eva_transform_mean = getattr(model_obj, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(model_obj, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            model_obj.image_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            model_obj.image_std = (eva_transform_std,) * 3

        return (model_obj,)


class ApplyPulid:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "pulid": ("PULID", ),
                "eva_clip": ("EVA_CLIP", ),
                "face_analysis": ("FACEANALYSIS", ),
                "image": ("IMAGE", ),
                "method": (["fidelity", "style", "neutral"],),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            },
            "optional": {
                "attn_mask": ("MASK", ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_pulid"
    CATEGORY = "pulid"

    def apply_pulid(self, model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None):
        work_model = model.clone()

        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        eva_clip.to(device, dtype=dtype)
        pulid_model = pulid.to(device, dtype=dtype)

        if attn_mask is not None:
            if attn_mask.dim() > 3:
                attn_mask = attn_mask.squeeze(-1)
            elif attn_mask.dim() < 3:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.to(device, dtype=dtype)

        if method == "fidelity" or projection == "ortho_v2":
            num_zero = 8
            ortho = False
            ortho_v2 = True
        elif method == "style" or projection == "ortho":
            num_zero = 16
            ortho = True
            ortho_v2 = False
        else:
            num_zero = 0
            ortho = False
            ortho_v2 = False

        if fidelity is not None:
            num_zero = fidelity

        image = tensor_to_image(image)

        # Init facexlib helper once if available — lazy, only for explicit face parsing
        if FACEXLIB_AVAILABLE:
            face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                device=device,
            )
            face_helper.face_parse = None
            face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device)
            bg_label = [0, 16, 18, 7, 8, 9, 14, 15]

        cond = []
        uncond = []

        for i in range(image.shape[0]):
            iface_embeds = None
            detected_face = None
            for det_size in [(s, s) for s in range(640, 256, -64)]:
                face_analysis.det_model.input_size = det_size
                faces = face_analysis.get(image[i])
                if faces:
                    detected_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                    iface_embeds = torch.from_numpy(detected_face.embedding).unsqueeze(0).to(device, dtype=dtype)
                    break

            if detected_face is None:
                print('Warning: No face detected in image', i)
                continue

            if FACEXLIB_AVAILABLE:
                face_helper.clean_all()
                face_helper.read_image(image[i])
                face_helper.get_face_landmarks_5(only_center_face=True)
                face_helper.align_warp_face()

                if len(face_helper.cropped_faces) == 0:
                    continue

                face_crop = face_helper.cropped_faces[0]
                face_t = image_to_tensor(face_crop).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                parsing_out = face_helper.face_parse(
                    T.functional.normalize(face_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                )[0]
                parsing_out = parsing_out.argmax(dim=1, keepdim=True)
                bg = sum(parsing_out == label for label in bg_label).bool()
                white_image = torch.ones_like(face_t)
                face_features_image = torch.where(bg, white_image, to_gray(face_t))
            else:
                # Fallback: affine-align via InsightFace keypoints, no background removal
                face_crop = _align_face_no_facexlib(image[i], detected_face)
                face_features_image = image_to_tensor(face_crop).unsqueeze(0).permute(0, 3, 1, 2).to(device)

            face_features_image = T.functional.resize(
                face_features_image, eva_clip.image_size,
                T.InterpolationMode.BICUBIC if 'cuda' in device.type else T.InterpolationMode.NEAREST
            ).to(device, dtype=dtype)
            face_features_image = T.functional.normalize(face_features_image, eva_clip.image_mean, eva_clip.image_std)

            id_cond_vit, id_vit_hidden = eva_clip(face_features_image, return_all_features=False, return_hidden=True, shuffle=False)
            id_cond_vit = id_cond_vit.to(device, dtype=dtype)
            for idx in range(len(id_vit_hidden)):
                id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

            id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

            id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)
            if noise == 0:
                id_uncond = torch.zeros_like(id_cond)
            else:
                id_uncond = torch.rand_like(id_cond) * noise
            id_vit_hidden_uncond = []
            for idx in range(len(id_vit_hidden)):
                if noise == 0:
                    id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[idx]))
                else:
                    id_vit_hidden_uncond.append(torch.rand_like(id_vit_hidden[idx]) * noise)

            cond.append(pulid_model.get_image_embeds(id_cond, id_vit_hidden))
            uncond.append(pulid_model.get_image_embeds(id_uncond, id_vit_hidden_uncond))

        if not cond:
            print("pulid warning: No faces detected in any of the given images, returning unmodified model.")
            return (work_model,)

        cond = torch.cat(cond).to(device, dtype=dtype)
        uncond = torch.cat(uncond).to(device, dtype=dtype)
        if cond.shape[0] > 1:
            cond = torch.mean(cond, dim=0, keepdim=True)
            uncond = torch.mean(uncond, dim=0, keepdim=True)

        if num_zero > 0:
            if noise == 0:
                zero_tensor = torch.zeros((cond.size(0), num_zero, cond.size(-1)), dtype=dtype, device=device)
            else:
                zero_tensor = torch.rand((cond.size(0), num_zero, cond.size(-1)), dtype=dtype, device=device) * noise
            cond = torch.cat([cond, zero_tensor], dim=1)
            uncond = torch.cat([uncond, zero_tensor], dim=1)

        sigma_start = work_model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = work_model.get_model_object("model_sampling").percent_to_sigma(end_at)

        patch_kwargs = {
            "pulid": pulid_model,
            "weight": weight,
            "cond": cond,
            "uncond": uncond,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "ortho": ortho,
            "ortho_v2": ortho_v2,
            "mask": attn_mask,
        }

        number = 0
        for id in [4,5,7,8]:
            block_indices = range(2) if id in [4, 5] else range(10)
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6):
            block_indices = range(2) if id in [3, 4, 5] else range(10)
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(work_model, patch_kwargs, ("middle", 1, index))
            number += 1

        return (work_model,)

class ApplyPulidAdvanced(ApplyPulid):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "pulid": ("PULID", ),
                "eva_clip": ("EVA_CLIP", ),
                "face_analysis": ("FACEANALYSIS", ),
                "image": ("IMAGE", ),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "projection": (["ortho_v2", "ortho", "none"],),
                "fidelity": ("INT", {"default": 8, "min": 0, "max": 32, "step": 1 }),
                "noise": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1 }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            },
            "optional": {
                "attn_mask": ("MASK", ),
            },
        }

NODE_CLASS_MAPPINGS = {
    "PulidModelLoader": PulidModelLoader,
    "PulidInsightFaceLoader": PulidInsightFaceLoader,
    "PulidEvaClipLoader": PulidEvaClipLoader,
    "ApplyPulid": ApplyPulid,
    "ApplyPulidAdvanced": ApplyPulidAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PulidModelLoader": "Load PuLID Model",
    "PulidInsightFaceLoader": "Load InsightFace (PuLID)",
    "PulidEvaClipLoader": "Load Eva Clip (PuLID)",
    "ApplyPulid": "Apply PuLID",
    "ApplyPulidAdvanced": "Apply PuLID Advanced",
}
