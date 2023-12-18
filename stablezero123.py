import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel
import numpy as np

#tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
class ImageSplit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "columns": ("INT", {
                    "default": 2, 
                    "min": 1, #Minimum value
                    "max": 100, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" 
                }),
                "lines": ("INT", {
                    "default": 3, 
                    "min": 1, #Minimum value
                    "max": 100, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" 
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "tests"

    def execute(self, images, columns, lines):
        image=images[0]
        i = 255. * image.cpu().numpy()
        pil_image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        imgwidth = pil_image.size[0]
        imgheight = pil_image.size[1]
        M=int(imgwidth/columns)
        N=int(imgheight/lines)
        
        tiles=[]
        for i in range(0, imgheight-imgheight%N, N):
            for j in range(0, imgwidth-imgwidth%M, M):
                box = (j, i, j+M, i+N)
                tiles.append(pil_image.crop(box))

        t_tiles=[]
        for tile in tiles:
            t_tile = tile.convert("RGB")
            t_tile = np.array(t_tile).astype(np.float32) / 255.0
            t_tile = torch.from_numpy(t_tile)[None,]
            t_tiles.append(t_tile)

        s=t_tiles[0]
        for i in range(1,len(t_tiles)):
            s = torch.cat((s, t_tiles[i]), dim=0)

        return (s,)


class Stablezero123:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "ckpt_name": ("STRING", {
                    "multiline": False,
                    "default": "sudo-ai/zero123plus-v1.1"
                }),
                "pipeline_name": ("STRING", {
                    "multiline": False,
                    "default": "sudo-ai/zero123plus-pipeline"
                }),
                "inference_steps": ("INT", {
                    "default": 75, 
                    "min": 1, #Minimum value
                    "max": 100, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" 
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "tests"

    def execute(self, images, ckpt_name, pipeline_name, inference_steps):
        pipeline = DiffusionPipeline.from_pretrained(ckpt_name, custom_pipeline=pipeline_name, torch_dtype=torch.float16)
        
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing')
        pipeline.to('cuda:0')
        
        image=images[0]
        i = 255. * image.cpu().numpy()
        cond = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        ##cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw)
        ## Converter para formato img

        image = pipeline(cond, num_inference_steps=inference_steps).images[0]
        
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)

class Stablezero123WithDepth:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "depth_images": ("IMAGE", ),
                "ckpt_name": ("STRING", {
                    "multiline": False,
                    "default": "sudo-ai/zero123plus-v1.1"
                }),
                "control_model_name": ("STRING", {
                    "multiline": False,
                    "default": "sudo-ai/controlnet-zp11-depth-v1"
                }),
                "pipeline_name": ("STRING", {
                    "multiline": False,
                    "default": "sudo-ai/zero123plus-pipeline"
                }),
                "inference_steps": ("INT", {
                    "default": 75, 
                    "min": 1, #Minimum value
                    "max": 100, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" 
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "tests"

    def execute(self, images, depth_images, ckpt_name, control_model_name, pipeline_name, inference_steps):
        pipeline = DiffusionPipeline.from_pretrained(ckpt_name, custom_pipeline=pipeline_name, torch_dtype=torch.float16)
        pipeline.add_controlnet(ControlNetModel.from_pretrained(control_model_name, torch_dtype=torch.float16), conditioning_scale=0.75)
        
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing')
        pipeline.to('cuda:0')
        
        image=images[0]
        i = 255. * image.cpu().numpy()
        cond = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        depth=depth_images[0]
        ii = 255. * depth.cpu().numpy()
        depth = Image.fromarray(np.clip(ii, 0, 255).astype(np.uint8))

        ##cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw)
        ## Converter para formato img

        image = pipeline(cond, depth_image=depth, num_inference_steps=inference_steps).images[0]
        
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    #"Stablezero123WithDepth" : Stablezero123WithDepth,
    "Stablezero123": Stablezero123,
    "SDZero ImageSplit" : ImageSplit
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    #"Stablezero123WithDepth": "Stablezero123WithDepth",
    "Stablezero123": "Stablezero123",
    "SDZero ImageSplit" : "SDZero ImageSplit"
}