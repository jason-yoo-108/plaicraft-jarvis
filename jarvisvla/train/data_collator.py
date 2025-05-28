'''
Author: Muyao 2350076251@qq.com
Date: 2025-03-04 23:26:27
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-05-28 23:20:36
'''


import numpy as np
import torch
from transformers import AutoProcessor
import pathlib 
import io
from rich import console
from typing import Literal,Union,Tuple
from rich import console
from copy import deepcopy
from typing import Callable
from PIL import Image, ImageEnhance
from torchvision import transforms
import random
import math
from typing import List, Tuple,Union
from pathlib import Path
from torch import Tensor
from copy import deepcopy

from jarvisvla.train.utils_train import IGNORE_TOKEN_ID

def make_collator(collator_type: str, **kwargs) -> Callable:
    """
    This function creates a data collator based on the specified collator type.
    
    Parameters:
        collator_type (str): The type of the data collator to be created.
        **kwargs: Additional keyword arguments to pass to the collator's constructor.
        
    Returns:
        Callable: A data collator instance corresponding to the given type.
    """
    collators = {
        "MultimodalChatDataCollatorforVLM": MultimodalChatDataCollatorforVLM,
        "VLAMultimodalChatDataCollatorforVLM": VLAMultimodalChatDataCollatorforVLM,
    }
    
    if collator_type in collators:
        # Instantiate the collator with the provided arguments
        return collators[collator_type](**kwargs)
    else:
        # Raise an error if the collator type is unknown
        raise ValueError(f"Unknown collator type: {collator_type}")

################
# MultimodalChatDataCollatorforVLM: Create a data collator to encode text and image pairs
################
class MultimodalChatDataCollatorforVLM:
    def __init__(self, processor:AutoProcessor, model_path:Union[pathlib.Path,str],  
                 max_seq_length:int = 1024, 
                 with_grounding:bool = True,
                 with_image:bool = True, resize_image:bool = True, image_folder:Union[pathlib.Path,str]=None,
                 random_image_size:Tuple[int]=(224,224),default_image_size:Tuple[int]=None,
                 image_factor:int=None, min_pixels:int=None,  max_pixels:int=None, max_ratio:int=None,
                 check:bool=False, get_length=False ):
        self.model_type = None
        self.processor = processor
        
        # tokenizer
        self.max_seq_length = max_seq_length
        self.user_template_token = None
        self.assistant_template_token = None
        self.user_template_token_len = None
        self.assistant_template_token_len = None
        self.tokenize_redundant = 0
        model_path = model_path.lower().replace('-','_')
        self.model_path = model_path
        
        
        # image
        self.image_folder = image_folder
        self.with_image = with_image
        self.random_image_size = random_image_size
        self.default_image_size = default_image_size
        self.resize_image = resize_image
        self.image_factor = image_factor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_ratio = max_ratio
        self.aug_methods = [
            DataAugment.HUE,
            DataAugment.SATURATION,
            DataAugment.BRIGHTNESS,
            DataAugment.CONTRAST,
            DataAugment.TRANSLATE
        ]
        
        # point
        self.with_grounding = with_grounding
        
        if "qwen2_vl" in model_path:
            self.model_type = "qwen2_vl"
            self.user_template_token = np.array([151644, 872],dtype=np.int64)#"<|im_start|>user\n"
            self.assistant_template_token = np.array([151644, 77091],dtype=np.int64)#"<|im_start|>assistant\n"
            self.tokenize_redundant = 0
            self.image_factor = 28
            self.min_pixels = processor.image_processor.min_pixels if min_pixels is None else min_pixels
            self.max_pixels = processor.image_processor.max_pixels if max_pixels is None else max_pixels
            self.max_ratio = 200
            self.resize_image = True
        else:
            ValueError(f"{model_path} is not support")
            
        self.user_template_token_len, self.assistant_template_token_len = len(self.user_template_token), len(self.assistant_template_token)
            
        self.check = check
        self.get_length = get_length

        if check:
            torch.set_printoptions(threshold=10000)
        self.my_console = console.Console()
    
    def __call__(self, examples):
        texts = []
        images = [] if self.with_image else None
        example_ids = []
        data_augment = DataAugment(
            model_type = self.model_type,
            methods=self.aug_methods,
            image_folder=self.image_folder,
            random_image_size=self.random_image_size,
            default_image_size=self.default_image_size,
            image_factor=self.image_factor,min_pixels=self.min_pixels,max_pixels=self.max_pixels,
            max_ratio=self.max_ratio
        )
        
        for edx,example in enumerate(examples):
            example_id = example.get("id")
            example_ids.append(example_id)
            conversations = example.get("conversations")
            data_augment.refresh()
            
            # prepare images
            if self.with_image:
                if "image_bytes" in example:
                    local_image_paths = example.get("image_bytes",[])
                else:
                    local_image_paths = example.get('image') if example.get('image') else []
                local_images = []
                current_visit_image_idx = 0
                if isinstance(local_image_paths, list):
                    pass
                elif isinstance(local_image_paths, str):
                    local_image_paths = [local_image_paths]
                else:
                    raise ValueError(f"{example_ids}, image must be a string or a list of strings.")
                
            
            for idx,conv in enumerate(conversations):
                sloved_contents = []
                for jdx,item in enumerate(conv["content"]):
                    item_type = item.get("type")
                    
                    # process images
                    if self.with_image and item_type == "image":
                        try:
                            image_path = local_image_paths[current_visit_image_idx]
                        except:
                            raise ValueError(f"{example_id}, Index out of range. Attempted to access index {current_visit_image_idx} in a list of size {len(local_image_paths)}.")
                        try:
                            image = data_augment.image_process(image_path)
                            local_images.append(image)
                            sloved_contents.append(item)
                        except Exception as e:
                            self.my_console.log(f"[red]{example_id}, can not process image. \n{e}")
                        current_visit_image_idx += 1
                    
                    # process points
                    elif self.with_grounding and item_type == "point":
                        new_item = {"type":"text"}
                        text = ""
                        caption = item.get("label","")
                        points = item.get("point",[])
                        points = [ data_augment.point_process(point) for point in points]
                        new_item["text"] = data_augment.create_point_prompt(points,caption)
                        sloved_contents.append(new_item)
                    elif self.with_grounding and item_type == "bbox":
                        new_item = {"type":"text"}
                        caption = item.get("label","")
                        bboxes = item.get("bbox",[])
                        bboxes = [ data_augment.bbox_process(bbox) for bbox in bboxes]
                        new_item["text"] = data_augment.create_bbox_prompt(bboxes,caption)
                        sloved_contents.append(new_item)
                    elif item_type == "text":
                        sloved_contents.append(item)
                    else:
                        raise ValueError(f"[red]{example_id},  type {item_type} is not support")
                        
                if not sloved_contents:
                    sloved_contents.append({"type":"text","text":"NULL"})
                    self.my_console.log(f"[blue]{example_id},[/blue][green] There's nothing left in this content")
                conv["content"] = sloved_contents
                
            assert current_visit_image_idx == len(local_image_paths), f"{example_id},  The number of images does not match."
            if self.with_image:
                images.extend(local_images)
   
            if "qwen2_vl" not in self.model_type:
                conversations = apply_private_conversations(conversations, self.model_type)
            text = self.processor.tokenizer.apply_chat_template(conversations,tokenize=False, add_generation_prompt=False)
            texts.append(text)
            
            if self.check:
                self.my_console.log(f"[blue]{example_id},[/blue][green] conversations: {conversations}")
                self.my_console.log(f"[green]~~~~~~~~~~~~~~~~~~~~~~")
                self.my_console.log(f"[blue]{example_id},[/blue][green] texts: {repr(texts[edx])}")
                self.my_console.log(f"[blue]{example_id},[/blue][green] texts: {texts[edx]}")
                if self.with_image and len(local_images) > 0:
                    self.my_console.log(f"[blue]{example_id},[/blue][green] "
                                        f"image num: {len(local_images)}"
                                        f"image shape: {local_images[0].shape}")
                self.my_console.log(f"[green]######################")
            
        if self.check or self.get_length:
            if images:
                batch_input_ids = self.processor(text = deepcopy(texts), images = deepcopy(images),)["input_ids"]
            else:
                batch_input_ids = self.processor.tokenizer(text = deepcopy(texts), )["input_ids"]
            
            batch_length_dict = {e["id"]:len(batch_input_id) for batch_input_id,e in zip(batch_input_ids,examples)}
            if self.check:
                self.my_console.log(f"[green]######################")
                self.my_console.log(f"[blue]batch_length_dict: {batch_length_dict}")
            else:
                return batch_length_dict
        
        # create batch
        if images:
            batch = self.processor(text = texts, images = images, return_tensors="pt", padding='max_length', max_length=self.max_seq_length, truncation=True)
        else:
            batch = self.processor.tokenizer(text = texts, return_tensors="pt", padding='max_length', max_length=self.max_seq_length, truncation=True)
        
        # check whether out of max token length
        labels = batch["input_ids"].clone()
        check_id = -1 if self.processor.tokenizer.padding_side=="right" else 0
        if labels[0][check_id].item()!=self.processor.tokenizer.pad_token_id:
            self.my_console.log("[red]Warning! the token length is probably out of max token length!")
        
        # mask user prompt
        for label in labels:
            np_label = label.cpu().numpy()
            label_len = len(label)
            beg_matches = np.where((np_label[np.arange(label_len - self.user_template_token_len + 1)[:, None] + np.arange(self.user_template_token_len)] == self.user_template_token).all(axis=1))[0].tolist()
            end_matches = np.where((np_label[np.arange(label_len - self.assistant_template_token_len + 1)[:, None] + np.arange(self.assistant_template_token_len)] == self.assistant_template_token).all(axis=1))[0].tolist()
            len_beg_matches = len(beg_matches)
            len_end_matches = len(end_matches)
            if not len_beg_matches:
                self.my_console.log(f"[red]Warning! len_beg_matches is {len_beg_matches}")
                continue
            # if out of max token length
            if len_beg_matches==len_end_matches+1:
                end_matches.append(self.max_seq_length)
                len_end_matches +=1
                self.my_console.log("[red]Warning! The final end match token for the user mask was not found.")
            assert len(beg_matches)==len(end_matches)
            
            try:
                label[:beg_matches[0]]= IGNORE_TOKEN_ID
                for instruction_beg_idx,instruction_end_idx in zip(beg_matches,end_matches):
                    label[instruction_beg_idx:instruction_end_idx]= IGNORE_TOKEN_ID
            except Exception as e:
                self.my_console.log(f"[red]Warning! {e} when `label[:beg_matches[0]]=-100`. Check the length: "
                        f"beg_matches length: {len(beg_matches)}, "
                        f"label length: {len(label)}")                
                raise e
            
        # mask padding token
        if self.processor.tokenizer.pad_token_id is not None:
            pad_mask = labels == self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id == self.processor.tokenizer.eos_token_id:
                pad_mask[:,1:]  = pad_mask[:,1:] & pad_mask[:,:-1]
            labels[pad_mask] = IGNORE_TOKEN_ID
            
        batch["labels"] = labels
        #import pdb; pdb.set_trace()
        return batch

class VLAMultimodalChatDataCollatorforVLM(MultimodalChatDataCollatorforVLM):
    def __init__(self, processor, model_path, **kwargs):
        super().__init__(processor=processor, model_path=model_path, **kwargs)
        self.aug_methods = [
            DataAugment.HUE,
            DataAugment.SATURATION,
            DataAugment.BRIGHTNESS,
            DataAugment.CONTRAST,
            DataAugment.TRANSLATE,
        ]

def apply_private_conversations(conversations:list, tokenizer=None):
    """Prepare the text from a sample of the dataset."""
    # LLAVA_next: batches处理的时候，输入的是所有images，然后根据<image>来分配
    conversations = []
    for conv in conversations:
        content = ""
        for item in conv["content"]:
            if item["type"] == "text":
                content += item["text"]
            elif item["type"] == "image":
                content += "<image>"
                image_count+=1
        conversations.append({"role": conv["role"], "content": content})
            
    return conversations

def image_hue_augmentation(image: Image.Image, hue_factor:float =  None,random_hue: float = 0.05) -> Image.Image:
    """Randomly adjust the hue of the image within a specified range."""
    if hue_factor is None:
        hue_factor = random.uniform(-random_hue, random_hue)
    return ImageEnhance.Color(image).enhance(1 + hue_factor)

def image_saturation_augmentation(image: Image.Image, saturation_factor:float = None, random_saturation: List[float] = [0.8, 1.2]) -> Image.Image:
    """Randomly adjust the saturation of the image within a specified range."""
    if saturation_factor is None:
        saturation_factor = random.uniform(*random_saturation)
    return ImageEnhance.Color(image).enhance(saturation_factor)

def image_brightness_augmentation(image: Image.Image, brightness_factor:float = None, random_brightness: List[float] = [0.8, 1.2]) -> Image.Image:
    """Randomly adjust the brightness of the image within a specified range."""
    if brightness_factor is None: 
        brightness_factor = random.uniform(*random_brightness)
    return ImageEnhance.Brightness(image).enhance(brightness_factor)

def image_contrast_augmentation(image: Image.Image, contrast_factor:float = None,  random_contrast: List[float] = [0.8, 1.2]) -> Image.Image:
    """Randomly adjust the contrast of the image within a specified range."""
    if contrast_factor is None: 
        contrast_factor = random.uniform(*random_contrast)
    return ImageEnhance.Contrast(image).enhance(contrast_factor)

def image_rotate_augmentation(image: Image.Image, rotate_degree: float = None, random_rotate: List[float] = [-2, 2]) -> Image.Image:
    """Randomly rotate the image within a specified degree range."""
    if rotate_degree is None:
        rotate_degree = random.uniform(*random_rotate)
    return image.rotate(rotate_degree, expand=False)

def image_scale_augmentation(image: Image.Image, x_scale_factor: float = None,y_scale_factor: float = None, scale_range: List[float] = [0.98, 1.02]) -> Image.Image:
    """Randomly scale the image by a factor within the given range."""
    if x_scale_factor is None:
        x_scale_factor = random.uniform(*scale_range)
        y_scale_factor = random.uniform(*scale_range)
    new_size = (int(image.width * x_scale_factor), int(image.height * y_scale_factor))
    return image.resize(new_size, Image.Resampling.LANCZOS)

def image_shear_augmentation(image: Image.Image, shear_degree: float = None, shear_range: float = 2) -> Image.Image:
    """Randomly shear the image using an affine transformation."""
    if shear_degree is None:
        shear_degree = random.uniform(-shear_range, shear_range)
    radians = shear_degree / 180 * 3.1415927
    return image.transform(image.size, Image.AFFINE, (1, -radians, 0, 0, 1, 0))

def image_flip_augmentation(image: Image.Image, x_flip: bool = None, y_flip: bool = None, flip_p = 0.02) -> Image.Image:
    """Randomly decide whether to flip the image horizontally or vertically using an affine transformation."""
    if x_flip is None:
        x_flip = random.choices([False, True], [1 - flip_p, flip_p])[0]
    if y_flip is None:
        y_flip = random.choices([False, True], [1 - flip_p, flip_p])[0]

    x_factor = -1 if x_flip else 1
    y_factor = -1 if y_flip else 1

    return image.transform(image.size, Image.AFFINE, 
         (x_factor, 0, 0 if x_factor == 1 else image.width, 0, y_factor, 0 if y_factor == 1 else image.height))

def image_translate_augmentation(image: Image.Image,trans_x:int = None,trans_y : int=None, max_trans: int = 2) -> Image.Image:
    """Randomly translate the image by a certain number of pixels.""" 
    if trans_x is None:
        trans_x = random.randint(-max_trans, max_trans)
    if trans_y is None:
        trans_y = random.randint(-max_trans, max_trans)
    return image.transform(image.size, Image.AFFINE, (1, 0, trans_x, 0, 1, trans_y))

def get_image_center(image_size: Tuple[float, float]):
    return (image_size[0] / 2, image_size[1] / 2)

def get_image_center(image_size: Tuple[float, float]) -> Tuple[float, float]:
    """示例中心点获取函数：返回图像中心 (cx, cy)。"""
    w, h = image_size
    return (w / 2.0, h / 2.0)

###################################

def point_rotate_augmentation(
    point: Tuple[float, float],
    image_size: Tuple[float, float],
    rotate_degree: float = None,
    expand: bool = False
):
    point_x, point_y = point
    w, h = image_size
    center = get_image_center(image_size)

    # 注意：rotate_degree > 0 时，这里的角度是顺时针旋转
    angle = math.radians(rotate_degree)
    
    # 构建基础仿射矩阵：先只考虑在原点旋转
    matrix = [
        round(math.cos(angle), 15),
        round(math.sin(angle), 15),
        0.0,
        round(-math.sin(angle), 15),
        round(math.cos(angle), 15),
        0.0,
    ]
    
    # transform 函数，用于将 (x, y) 应用矩阵变换
    def transform(x, y, matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f
    
    # 第一步：将图像从 (cx, cy) 平移到原点，再做旋转
    # 也就是先对 -(center_x), -(center_y) 做 transform，然后再回移回来
    matrix[2], matrix[5] = transform(-center[0], -center[1], matrix)
    matrix[2] += center[0]
    matrix[5] += center[1]
    
    # 如果需要 expand，重新计算旋转后包裹图像的新的 w, h，并对 matrix 进行平移修正
    # TODO 有bug 暂时无法启用
    if expand:
        # calculate output size
        xx = []
        yy = []
        for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
            x, y = transform(x, y, matrix)
            xx.append(x)
            yy.append(y)
        nw = math.ceil(max(xx)) - math.floor(min(xx))
        nh = math.ceil(max(yy)) - math.floor(min(yy))

        # We multiply a translation matrix from the right.  Because of its
        # special form, this is the same as taking the image of the
        # translation vector as new translation vector.
        matrix[2], matrix[5] = transform(-(nw - w) / 2.0, -(nh - h) / 2.0, matrix)
        w, h = nw, nh
    
    # 最后一步：对目标点进行矩阵变换，得到旋转后的 new_x, new_y
    new_x, new_y = transform(point_x, point_y, matrix)
    return (new_x, new_y), (w, h)

def point_scale_augmentation(point: Tuple[float, float],  image_size: Tuple[float, float], x_scale_factor: float = 1.0, y_scale_factor: float = 1.0):
    x, y = point
    new_width = image_size[0] * x_scale_factor
    new_height = image_size[1] * y_scale_factor
    x = x * x_scale_factor
    y = y * y_scale_factor
    #print(scale_factor, x , y,new_width,new_height)
    return (x, y), (new_width, new_height)

def point_shear_augmentation(point: Tuple[float, float], shear_degree: float = 0.0):
    x, y = point
    radians = math.radians(shear_degree)
    x_new = x + y * math.tan(radians)
    y_new = y  # y remains unchanged
    return (x_new, y_new)

def point_flip_augmentation(point: Tuple[float, float], image_size: Tuple[int, int], x_flip: bool = False, y_flip: bool = False):
    x, y = point
    width, height = image_size
    if x_flip:
        x = width - x
    if y_flip:
        y = height - y
    return (x, y)

def point_translate_augmentation(point: Tuple[float, float], trans_x: float = 0.0, trans_y: float = 0.0):
    x, y = point
    x_new = x - trans_x
    y_new = y - trans_y
    return (x_new, y_new)

def point_resize(point: Tuple[float, float], from_size: Tuple[float, float], to_size: Tuple[float, float]):
    x, y = point
    # Correct the scale factors
    x_scale_factor = to_size[0] / from_size[0]
    y_scale_factor = to_size[1] / from_size[1]
    # Apply the scale factors
    x = x * x_scale_factor
    y = y * y_scale_factor
    return (x, y)

#############################

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int,max_ratio:int
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def fetch_image(image: Image.Image,  factor: int, min_pixels: int, max_pixels: int,max_ratio:int) -> Image.Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_ratio=max_ratio,
        )
    image = image.resize((resized_width, resized_height))
    return image

#############################

class DataAugment(object):
    # 定义可能使用到的增强方法名称
    HUE = "hue"
    SATURATION = "saturation"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    ROTATE = "rotate"
    SCALE = "scale"
    SHEAR = "shear"
    FLIP = "flip"
    TRANSLATE = "translate"
    LEGAL_METHODS = {HUE,SATURATION,BRIGHTNESS,CONTRAST,ROTATE,SCALE,SHEAR,FLIP,TRANSLATE}
    
    def __init__(self, model_type:str, methods: List[str],
                 image_folder:Union[Path,str]=None,
                 random_image_size: Tuple[int, int]=(224,224), default_image_size: Tuple[int, int]=None,
                 image_factor:int=None, min_pixels:int=None,  max_pixels:int=None, max_ratio:int=None):
        """
        methods: 传入想要应用的增强类型列表, 比如 [DataAugment.ROTATE, DataAugment.TRANSLATE, ...]
        """
        
        self.model_type = model_type
        
        self.methods = methods  
        for method in methods:
            assert method in DataAugment.LEGAL_METHODS, f"illegal method: {method}"
        self.params = {} # 用于存储当次刷新得到的随机参数
        
        self.image_folder = image_folder
        self.random_image_size = random_image_size
        self.default_image_size = default_image_size
        self.image_factor = image_factor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_ratio = max_ratio
        
        self.raw_image_size = None
        self.augment_image_size = None
        self.resize_image_size = None
        
    def refresh(self,
        hue_range: float = 0.05,
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        rotate_range: Tuple[float, float] = (-5, 5),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        shear_range: float = 2,
        flip_p: float = 0.05,
        translate_max: int = 10
        ):
        """
        refresh() 用来刷新一次「本批次」的增强参数，比如随机多少度旋转、平移多少像素等等。
        每次调用 refresh()，都会覆盖以前的参数。
        """
        self.params = {}  # 每次重新生成

        # 颜色类增强
        if self.HUE in self.methods:
            # hue_factor 在 -hue_range ~ +hue_range 之间
            hue_factor = random.uniform(-hue_range, hue_range)
            self.params[self.HUE] = hue_factor

        if self.SATURATION in self.methods:
            saturation_factor = random.uniform(*saturation_range)
            self.params[self.SATURATION] = saturation_factor

        if self.BRIGHTNESS in self.methods:
            brightness_factor = random.uniform(*brightness_range)
            self.params[self.BRIGHTNESS] = brightness_factor

        if self.CONTRAST in self.methods:
            contrast_factor = random.uniform(*contrast_range)
            self.params[self.CONTRAST] = contrast_factor

        # 几何类增强
        if self.ROTATE in self.methods:
            rotate_degree = random.uniform(*rotate_range)
            self.params[self.ROTATE] = rotate_degree
        
        if self.SCALE in self.methods:
            x_scale_factor = random.uniform(*scale_range)
            y_scale_factor = random.uniform(*scale_range)
            self.params[self.SCALE] = (x_scale_factor,y_scale_factor)
        
        if self.SHEAR in self.methods:
            shear_degree = random.uniform(-shear_range, shear_range)
            self.params[self.SHEAR] = shear_degree
        
        if self.FLIP in self.methods:
            # 随机决定是否水平翻转、是否垂直翻转
            x_flip = random.choices([False, True], [1 - flip_p, flip_p])[0]
            y_flip = random.choices([False, True], [1 - flip_p, flip_p])[0]
            self.params[self.FLIP] = (x_flip, y_flip)
        
        if self.TRANSLATE in self.methods:
            trans_x = random.randint(-translate_max, translate_max)
            trans_y = random.randint(-translate_max, translate_max)
            self.params[self.TRANSLATE] = (trans_x, trans_y)
           
    def image_open(self,image_input:Union[str,Path]) -> Image.Image:
        if isinstance(image_input, bytes):
            # Handle the input as image bytes
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, (str, Path)):
            # Normalize to Path if it's a string
            if isinstance(image_input, str):
                image_input = Path(image_input)

            # If not an absolute path, assume it's relative to the image folder
            if not image_input.is_absolute():
                image_input = self.image_folder / image_input

            # Check if the image file exists
            if not image_input.exists():
                raise ValueError(f"Image file {image_input} not found.")

            # Open the image from the filesystem
            image = Image.open(image_input).convert("RGB")
        else:
            # If input isn't bytes, str, or Path, raise an error
            raise TypeError("image_input must be a path (str or Path) or bytes")
        
        return image
            
    def image_augment(self, image: Image.Image) -> Image.Image:
        """ 根据 flesh() 刷新的参数，对图像进行增强。 """
        for method in self.methods:
            if method == self.HUE:
                image = image_hue_augmentation(image,hue_factor=self.params[self.HUE])
                
            elif method == self.SATURATION:
                image = image_saturation_augmentation(image,saturation_factor=self.params[self.SATURATION])
                
            elif method == self.BRIGHTNESS:
                image = image_brightness_augmentation(image,brightness_factor=self.params[self.BRIGHTNESS])
                
            elif method == self.CONTRAST:
                image = image_contrast_augmentation(image, contrast_factor=self.params[self.CONTRAST])
                
            elif method == self.ROTATE:
                image = image_rotate_augmentation(image,rotate_degree=self.params[self.ROTATE])
            
            elif method == self.SCALE:
                x_scale_factor,y_scale_factor = self.params[self.SCALE]
                image = image_scale_augmentation(image, x_scale_factor=x_scale_factor,y_scale_factor=y_scale_factor )
                
            elif method == self.SHEAR:
                image = image_shear_augmentation(image, shear_degree=self.params[self.SHEAR])
                
            elif method == self.FLIP:
                x_flip, y_flip = self.params[self.FLIP]
                image = image_flip_augmentation(image, x_flip=x_flip,y_flip=y_flip)
            
            elif method == self.TRANSLATE:
                trans_x, trans_y = self.params[self.TRANSLATE]
                image = image_translate_augmentation(image, trans_x=trans_x, trans_y=trans_y)
            
        return image
    
    def image_resize(self,image: Image.Image) -> Image.Image:
        if "qwen2_vl" in self.model_type:
            image = fetch_image(image,factor=self.image_factor,min_pixels=self.min_pixels,max_pixels=self.max_pixels,max_ratio=self.max_ratio,)
        return image
        
    def image_process(self, image_path:Union[str,Path]) -> Tensor:
        
        try:  
            # 打开并统一转换到RGB
            image = self.image_open(image_path)
            self.raw_image_size = image.size
            
            # image数据增强
            image = self.image_augment(image)
            self.augment_image_size = image.size
            
            # 缩放到VLM适合的大小
            image = self.image_resize(image)
            self.resize_image_size = image.size
            
            #1. 从整数（通常范围是 0-255）转换为浮点数。
            #2. 将像素值缩放到 [0, 1] 的范围。
            #3. 维度重排序
            transform = transforms.ToTensor() 
            image = transform(image)
            
        except Exception as e:
            self.my_console.log(f"{e}")
            self.my_console.log(f"[red]can't open {image_path} correctly")
            raise Exception(f"{e}. Can't open {image_path} correctly")
        return image
          
    def point_adapt(self, point: Tuple[float, float]):
        """
        point: (x1, y1), (x2, y2)
        适应图像经过变换后的新点坐标列表。
        """
        # 特对于 rotate / scale / shear / flip / translate，要与图像同一中心或同一个坐标变换基准。
        image_size = deepcopy(self.raw_image_size)
        width, height = image_size
        
        x = point[0]/100 * width
        y = point[1]/100 * height
        point = (x, y)

        for method in self.methods:
            # 1) ROTATE
            if method == self.ROTATE:
                point,image_size = point_rotate_augmentation(point,image_size,rotate_degree = self.params[self.ROTATE])
            
            # 2) SCALE
            elif method == self.SCALE:
                x_scale_factor,y_scale_factor = self.params[self.SCALE]
                point,image_size = point_scale_augmentation(point,image_size, x_scale_factor=x_scale_factor,y_scale_factor=y_scale_factor,)

            # 3) SHEAR
            elif method == self.SHEAR:
                point = point_shear_augmentation(point,self.params[self.SHEAR])

            # 4) FLIP
            elif method == self.FLIP:
                x_flip, y_flip = self.params[self.FLIP]
                point = point_flip_augmentation(point,image_size,x_flip,y_flip)
                
            # 5) TRANSLATE
            elif method == self.TRANSLATE:
                trans_x, trans_y = self.params[self.TRANSLATE]
                point = point_translate_augmentation(point,trans_x,trans_y)
        
        width, height = self.augment_image_size
        x = point[0] / width * 100
        y = point[1] / height * 100
        
        return (x,y)

    def point_augment(self, point: Tuple[float, float],
                ) -> List[Tuple[float, float]]:
        
        w,h = deepcopy(self.resize_image_size)
        x = point[0] / 100 * w
        y = point[1] / 100 * h
        x_distract_pixel = random.uniform(-1,1)
        y_distract_pixel = random.uniform(-1,1)
        x += x_distract_pixel
        y += y_distract_pixel
        x = point[0] / w * 100
        y = point[1] / h * 100
        return (x,y)
    
    def add_point_template(self, point: Tuple[float, float]):
        output_point = []
        if self.model_type=="qwen2_vl":
            x = 99 if point[0] >= 100 else point[0]
            y = 99 if point[1] >= 100 else point[1]
            output_point = [int(x * 10),int(y * 10)]
        return output_point
    
    def point_process(self, point: Tuple[float, float]):
        point = self.point_adapt(point)
        point = self.point_augment(point)
        return self.add_point_template(point)

def point_with_guide(image, points, guide=[]):
    import numpy as np
    import cv2

    if isinstance(image, Image.Image):
        img_array = np.array(image)
        cv2_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif isinstance(image, str):
        cv2_image = cv2.imread(image)

    if not points:
        return cv2_image

    height, width = cv2_image.shape[:2]
    color = (255, 105, 180)  # Pink color in BGR format
    guide_color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    radius = 2
    guide_radius = 5

    if guide:
        cv2.circle(cv2_image, (guide[0], guide[1]), guide_radius, guide_color, -1)

    for point in points:
        x, y, text = point
        x = int(x * width)
        y = int(y * height)
        cv2.circle(cv2_image, (x, y), radius, color, -1)
        cv2.putText(cv2_image, text, (x + 15, y), font, font_scale, color, thickness)

    cv2.imwrite("1.jpg", cv2_image)
    return cv2_image
            
