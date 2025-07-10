'''
Author: Muyao 2350076251@qq.com
Date: 2025-02-18 15:57:29
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-05 22:48:27
'''

import re
import numpy as np
import copy
from collections import OrderedDict
from typing import Union, List, Dict, Tuple
import torch
from rich import console
from abc import ABC, abstractmethod

from minestudio.utils.vpt_lib.actions import ActionTransformer, Buttons
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.simulator.entry import CameraConfig


def tag_token(place: int, tokenizer_type: str, return_type: int = 0):
    """
    Return the start or end tag token based on the position.

    Args:
        place (int): 0 for the start tag, 1 for the end tag.
        tokenizer_type (str): Tokenizer type;.
        return_type (int): Specifies which part of the token to return: 0 for token text, 1 for numeric identifier.

    Returns:
        tuple: (token text, token numeric identifier)
    """
    assert place in {0, 1}
    if tokenizer_type == "qwen2_vl":
        special_tokens = [
            ('<|reserved_special_token_178|>', 151835),
            ('<|reserved_special_token_179|>', 151836),
        ]
    else:
        raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
    return special_tokens[place][return_type]


class ActionTokenizer(ABC):
    """
    Base class for action tokenizers, used to encode and decode actions to and from tokens.
    """
    # Define common movement and operation actions
    movements = ('forward', 'back', 'left', 'right', 'sprint', 'sneak')
    operations = ('use', 'drop', 'attack', 'jump')

    def __init__(self,
                 tokenizer_type="qwen2_vl",
                 camera_quantization_scheme="mu_law",
                 camera_mu=20,
                 camera_binsize=1,
                 camera_maxval=10):
        self.tokenizer_type = tokenizer_type

        # Retrieve the start and end tag tokens and their IDs
        self.act_beg_id = tag_token(0, self.tokenizer_type, return_type=1)
        self.act_end_id = tag_token(1, self.tokenizer_type, return_type=1)
        self.act_beg_token = tag_token(0, self.tokenizer_type, return_type=0)
        self.act_end_token = tag_token(1, self.tokenizer_type, return_type=0)

        # Initialize camera configuration
        camera_config = CameraConfig(
            camera_maxval=camera_maxval,
            camera_binsize=camera_binsize,
            camera_quantization_scheme=camera_quantization_scheme,
            camera_mu=camera_mu,
        )
        self.n_camera_bins = camera_config.n_camera_bins

        # Define the null action with default values (False for buttons, (0.0, 0.0) for camera)
        self.null_action = {
            'forward': False, 'back': False, 'left': False, 'right': False,
            'sprint': False, 'sneak': False,
            'hotbar.1': False, 'hotbar.2': False, 'hotbar.3': False, 'hotbar.4': False,
            'hotbar.5': False, 'hotbar.6': False, 'hotbar.7': False, 'hotbar.8': False, 'hotbar.9': False,
            'use': False, 'drop': False, 'attack': False, 'jump': False,
            'inventory': False,
            'camera': (0.0, 0.0)
        }

        # Initialize action transformer and action mapper
        self.action_transformer = ActionTransformer(**camera_config.action_transformer_kwargs)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=camera_config.n_camera_bins)

    @abstractmethod
    def encode(self, actions: Dict) -> Union[torch.Tensor, list, str]:
        """
        Abstract method: Encode actions into tokens.

        Args:
            actions (Dict): Dictionary of actions.

        Returns:
            Union[torch.Tensor, list, str]: Encoded token representation.
        """
        pass

    @abstractmethod
    def decode(self, tokens: Union[torch.Tensor, list]) -> List[OrderedDict]:
        """
        Abstract method: Decode tokens into actions.

        Args:
            tokens (Union[torch.Tensor, list]): Sequence of tokens (string type is not allowed).

        Returns:
            List[OrderedDict]: A list of decoded actions as OrderedDict objects.
        """
        pass


"""
JARVIS_PLAICRAFT_MAP = {
    # NOTE: Key / mouse presses
    151838: ("key_press", "1"), 151839: ("key_press", "2"), 151840: ("key_press", "3"), 151841: ("key_press", "4"),
    151842: ("key_press", "5"), 151843: ("key_press", "6"), 151844: ("key_press", "7"), 151845: ("key_press", "8"), 151846: ("key_press", "9"),
    151848: ("key_press", "w"), 151849: ("key_press", "s"),
    151851: ("key_press", "a"), 151852: ("key_press", "d"),
    151854: ("key_press", "Control_L", "w"), 151855: ("key_press", "Shift"),
    151857: ("mouse_press", "mouse_right"),  # NOTE: 151857 is use
    151859: ("key_press", "q"),
    151861: ("mouse_press", "mouse_left"),   # NOTE: 151861 is attack
    151863: ("key_press", "space"),
    151834: ("key_press", "e"),
    # NOTE: Camera pitch quantization
    151866: ("mouse_x", -150), 151867: ("mouse_x", -135), 151868: ("mouse_x", -120), 151869: ("mouse_x", -105), 151870: ("mouse_x", -90),
    151871: ("mouse_x", -75), 151872: ("mouse_x", -60), 151873: ("mouse_x", -45), 151874: ("mouse_x", -30), 151875: ("mouse_x", -15),
    151876: ("mouse_x", 0), 151877: ("mouse_x", 15), 151878: ("mouse_x", 30), 151879: ("mouse_x", 45), 151880: ("mouse_x", 60),
    151881: ("mouse_x", 75), 151882: ("mouse_x", 90), 151883: ("mouse_x", 105), 151884: ("mouse_x", 120), 151885: ("mouse_x", 135), 151886: ("mouse_x", 150),
    # NOTE: Camera yaw quantization
    151887: ("mouse_y", -100), 151888: ("mouse_y", -90), 151889: ("mouse_y", -80), 151890: ("mouse_y", -70), 151891: ("mouse_y", -60),
    151892: ("mouse_y", -50), 151893: ("mouse_y", -40), 151894: ("mouse_y", -30), 151895: ("mouse_y", -20), 151896: ("mouse_y", -10),
    151897: ("mouse_y", 0), 151898: ("mouse_y", 10), 151899: ("mouse_y", 20), 151900: ("mouse_y", 30), 151901: ("mouse_y", 40),
    151902: ("mouse_y", 50), 151903: ("mouse_y", 60), 151904: ("mouse_y", 70), 151905: ("mouse_y", 80), 151906: ("mouse_y", 90), 151907: ("mouse_y", 100),
}
"""


print("Reloaded!")

ACTION_BEGIN_TOKEN = "<|reserved_special_token_178|>"
ACTION_END_TOKEN = "<|reserved_special_token_179|>"
PLAICRAFT_TO_JARVIS_MAP = {
    # NOTE: Keyboard and mouse presses
    ("mouse_press", "mouse_left"):   "<|reserved_special_token_204|>",
    ("mouse_press", "mouse_right"):  "<|reserved_special_token_200|>",
    ("key_press", "1"):              "<|reserved_special_token_181|>",
    ("key_press", "2"):              "<|reserved_special_token_182|>",
    ("key_press", "3"):              "<|reserved_special_token_183|>",
    ("key_press", "4"):              "<|reserved_special_token_184|>",
    ("key_press", "5"):              "<|reserved_special_token_185|>",
    ("key_press", "6"):              "<|reserved_special_token_186|>",
    ("key_press", "7"):              "<|reserved_special_token_187|>",
    ("key_press", "8"):              "<|reserved_special_token_188|>",
    ("key_press", "9"):              "<|reserved_special_token_189|>",
    ("key_press", "w"):              "<|reserved_special_token_191|>",
    ("key_press", "s"):              "<|reserved_special_token_192|>",
    ("key_press", "a"):              "<|reserved_special_token_194|>",
    ("key_press", "d"):              "<|reserved_special_token_195|>",
    ("key_press", "Control_L", "w"): "<|reserved_special_token_197|>",
    # ("key_press", "Control_R", "w"): "<|reserved_special_token_197|>",
    ("key_press", "Shift"):          "<|reserved_special_token_198|>",
    ("key_press", "q"):              "<|reserved_special_token_202|>",
    ("key_press", "space"):          "<|reserved_special_token_206|>",
    ("key_press", "e"):              "<|reserved_special_token_177|>",
    # NOTE: Camera pitch quantization
    ("mouse_x", -150):               "<|reserved_special_token_209|>",
    ("mouse_x", -135):               "<|reserved_special_token_210|>",
    ("mouse_x", -120):               "<|reserved_special_token_211|>",
    ("mouse_x", -105):               "<|reserved_special_token_212|>",
    ("mouse_x", -90):                "<|reserved_special_token_213|>",
    ("mouse_x", -75):                "<|reserved_special_token_214|>",
    ("mouse_x", -60):                "<|reserved_special_token_215|>",
    ("mouse_x", -45):                "<|reserved_special_token_216|>",
    ("mouse_x", -30):                "<|reserved_special_token_217|>",
    ("mouse_x", -15):                "<|reserved_special_token_218|>",
    ("mouse_x", 0):                  "<|reserved_special_token_219|>",
    ("mouse_x", 15):                 "<|reserved_special_token_220|>",
    ("mouse_x", 30):                 "<|reserved_special_token_221|>",
    ("mouse_x", 45):                 "<|reserved_special_token_222|>",
    ("mouse_x", 60):                 "<|reserved_special_token_223|>",
    ("mouse_x", 75):                 "<|reserved_special_token_224|>",
    ("mouse_x", 90):                 "<|reserved_special_token_225|>",
    ("mouse_x", 105):                "<|reserved_special_token_226|>",
    ("mouse_x", 120):                "<|reserved_special_token_227|>",
    ("mouse_x", 135):                "<|reserved_special_token_228|>",
    ("mouse_x", 150):                "<|reserved_special_token_229|>",
    # NOTE: Camera yaw quantization
    ("mouse_y", -100):               "<|reserved_special_token_230|>",
    ("mouse_y", -90):                "<|reserved_special_token_231|>",
    ("mouse_y", -80):                "<|reserved_special_token_232|>",
    ("mouse_y", -70):                "<|reserved_special_token_233|>",
    ("mouse_y", -60):                "<|reserved_special_token_234|>",
    ("mouse_y", -50):                "<|reserved_special_token_235|>",
    ("mouse_y", -40):                "<|reserved_special_token_236|>",
    ("mouse_y", -30):                "<|reserved_special_token_237|>",
    ("mouse_y", -20):                "<|reserved_special_token_238|>",
    ("mouse_y", -10):                "<|reserved_special_token_239|>",
    ("mouse_y", 0):                  "<|reserved_special_token_240|>",
    ("mouse_y", 10):                 "<|reserved_special_token_241|>",
    ("mouse_y", 20):                 "<|reserved_special_token_242|>",
    ("mouse_y", 30):                 "<|reserved_special_token_243|>",
    ("mouse_y", 40):                 "<|reserved_special_token_244|>",
    ("mouse_y", 50):                 "<|reserved_special_token_245|>",
    ("mouse_y", 60):                 "<|reserved_special_token_246|>",
    ("mouse_y", 70):                 "<|reserved_special_token_247|>",
    ("mouse_y", 80):                 "<|reserved_special_token_248|>",
    ("mouse_y", 90):                 "<|reserved_special_token_249|>",
    ("mouse_y", 100):                "<|reserved_special_token_250|>",
}
JARVIS_TO_PLAICRAFT_MAP = {v: k for k, v in PLAICRAFT_TO_JARVIS_MAP.items()}


class PlaicraftActionTokenizer(ActionTokenizer):
    """
    Single action tokenizer that implements the specific encoding and decoding logic.

    BUTTONS_GROUPS:
        Names of different action groups.
    """
    BUTTONS_GROUPS = [
        "hotbar", "fore or back", "left or right", "sprint or sneak", "use",
        "drop", "attack", "jump", "camera"
    ]

    def __init__(self,
                 tokenizer_type="qwen2_vl",
                 bases: list = [10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 21, 21],
                 camera_quantization_scheme="mu_law",
                 camera_mu=20,
                 camera_binsize=1):
        # Call the parent constructor to initialize common configurations
        super().__init__(tokenizer_type=tokenizer_type,
                         camera_quantization_scheme=camera_quantization_scheme,
                         camera_mu=camera_mu,
                         camera_binsize=camera_binsize)
        # Log related information using rich console
        console.Console().log(f"tokenizer_type: {tokenizer_type}")
        console.Console().log(f"bases: {bases}, camera_mu: {camera_mu}, n_camera_bins: {self.n_camera_bins}, camera_binsize: {camera_binsize}")
        self.bases = bases
        # NULL_ACTION is the default null action; its encoding uses the middle values of the last two elements of bases
        self.NULL_ACTION = [0, (bases[-2] // 2) * bases[-2] + (bases[-1] // 2)]
    
    def decode(self, tokens: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Given a string that contains a list of tokens, return PLAICraft keyboard and mouse presses as well as mouse movements.
        """
        # assert len(tokens) == 1, "JarvisVLA returns one output token sequence strings, not multiple."
        tokens = re.findall(r'<\|reserved_special_token_\d+\|>', tokens)
        assert len(tokens) > 0, "Must have received at least 1 valid token."

        key_presses, mouse_movements = [], []
        curr_mouse_movement = [None, None]
        for token in tokens:
            info = JARVIS_TO_PLAICRAFT_MAP.get(token, ("unknown", None))
            if info[0] == "key_press":
                key_presses.extend(info[1:])
            elif info[0] == "mouse_press":
                if info[1] == "mouse_left":
                    key_presses.extend(info[1:])
                elif info[1] == "mouse_right":
                    key_presses.extend(info[1:])
            elif info[0] in ["mouse_x", "mouse_y"]:
                if info[0] == "mouse_x":
                    assert curr_mouse_movement[0] is None, "Mouse X movement already set."
                    curr_mouse_movement[0] = info[1]
                    if curr_mouse_movement[1] is not None:
                        mouse_movements.append(tuple(curr_mouse_movement))
                        curr_mouse_movement = [None, None]
                else:
                    assert curr_mouse_movement[1] is None, "Mouse Y movement already set."
                    curr_mouse_movement[1] = info[1]
                    if curr_mouse_movement[0] is not None:
                        mouse_movements.append(tuple(curr_mouse_movement))
                        curr_mouse_movement = [None, None]
        return list(set(key_presses)), mouse_movements

    def encode(self, key_presses: List[str], mouse_movements: List[str]) -> str:
        """
        Given a list of PLAICraft keyboard and mouse presses as well as mouse movements.
        """
        result = []
        for key_press in key_presses:
            token = PLAICRAFT_TO_JARVIS_MAP.get(("key_press", key_press), None)
            if token is None:
                print(f"Warning: Key press '{key_press}' not found in PLAICraft to JARVIS mapping.")
            else:
                result.append(token)
        tt=1
        for mouse_movement in mouse_movements:
            pitch, yaw = mouse_movement[0], mouse_movement[1]
            pitch_match, yaw_match, pitch_min_dist, yaw_min_dist = None, None, float('inf'), float('inf')
            for key_info in PLAICRAFT_TO_JARVIS_MAP.keys():
                key_type, value = key_info[0], key_info[1]
                if key_type == "mouse_x" and abs(value - pitch) < pitch_min_dist:
                    pitch_match, pitch_min_dist = value, abs(value - pitch)
                elif key_type == "mouse_y" and abs(value - yaw) < yaw_min_dist:
                    yaw_match, yaw_min_dist = value, abs(value - yaw)
            result.extend((PLAICRAFT_TO_JARVIS_MAP[("mouse_x", pitch_match)], PLAICRAFT_TO_JARVIS_MAP[("mouse_y", yaw_match)]))
            print(tt, result)
            tt+=1
        # print("FINAL", "".join([ACTION_BEGIN_TOKEN]+result+[ACTION_END_TOKEN]))
        return "".join([ACTION_BEGIN_TOKEN]+result+[ACTION_END_TOKEN])

    def encode_action(self, action: tuple) -> str:
        """
        Encode a single action into a control token string.

        Args:
            action (tuple): A tuple (buttons, camera).

        Returns:
            str: The encoded control token string.
        """
        raise NotImplementedError("The encode_action method is not implemented in PlaicraftActionTokenizer.")

    def group_action_2_token(self, group_action):
        """
        Convert a group action representation into a control token string.

        Args:
            group_action: A list of numbers representing each part of the action.

        Returns:
            str: The concatenated control token string (with start and end tags).
        """
        raise NotImplementedError("The group_action_2_token method is not implemented in PlaicraftActionTokenizer.")

    def decimal_action_2_group_action(self, inputs: tuple):
        """
        Convert a decimal action representation into a group action representation with varying bases.

        Args:
            inputs (tuple): A tuple of two decimal integers representing button and camera actions.

        Returns:
            tuple: Each element represents the value for one action group.

        Description:
            - For button actions, perform successive modulo and integer division operations according to the bases.
            - If the button part equals 8640, mark it as inventory mode and set it to 0.
            - For camera actions, process the last two parts separately.
        """
        raise NotImplementedError("decimal_action_2_group_action method is not implemented in PlaicraftActionTokenizer.")

    def null_token(self) -> str:
        """
        Get the token corresponding to the null action.

        Returns:
            str: The control token string for the null action.
        """
        raise NotImplementedError("The null_token method is not implemented in PlaicraftActionTokenizer.")
