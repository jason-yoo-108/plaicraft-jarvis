import time

from rich import print
from openai import OpenAI
import random
from typing import Literal
import copy
from pathlib import Path
from collections import Counter
import numpy as np

from jarvisvla.inference import action_mapping, load_model, processor_wrapper
from jarvisvla.utils.file_utils import load_json_file
from mappers import PlaicraftActionTokenizer

#################
# prompt
#################

INSTRUCTION_TEMPLATE = [
    "help me to craft a {}.",
    "craft a {}.",
    "Craft a {}.",
    "Could you craft a {} for me?",
    "I need you to craft a {}.",
    "Please craft a {} in the game.",
    "Craft me a {} quickly.",
    "Make sure to craft a {} for the task.",
    "Craft a {} so I can use it.",
    "Let’s craft a {} for this project.",
    "I need you to craft {} right now.",
]


class PlaicraftAgent:
    def __init__(self,checkpoint_path, base_url, api_key="EMPTY",
                 history_num=0,action_chunk_len=1, bpe=0,
                 instruction_type:Literal['simple','recipe','normal'] = 'normal',
                 temperature=0.5):
        
        self.LLM_backbone,self.VLM_backbone = load_model.load_visual_model(checkpoint_path=checkpoint_path)
        self.action_tokenizer = PlaicraftActionTokenizer()
        
        self.prompt_library = load_json_file(Path(__file__).parent/"assets"/"instructions.json") #存储我写好的instructions
        self.recipe_fold=Path(__file__).parent/"assets"/"recipes" # 存储所有recipes的文件夹
        self.recipes = dict()  #制作方案集合
        self.method_map = {
            True: "crafting table",
            False: "inventory",
        }
        
        self.processor_wrapper = processor_wrapper.ProcessorWrapper(None,model_name=self.VLM_backbone)
       
        self.actions = []
        self.action_chunk_len=action_chunk_len  # 一次返回一个action chunk
        # 用于带有记忆的agent
        self.history_num = history_num
        self.history = []
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        models = self.client.models.list()
        self.model = models.data[0].id
        
        self.temperature = temperature
        self.instruction_type = instruction_type
        
        self.tokenizer = None
        from transformers import AutoTokenizer
        if self.LLM_backbone in {"llama-3","llama-2","qwen2_vl"}:
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,  
                trust_remote_code=True,
            )
            if self.LLM_backbone=="qwen2_vl" and len(self.tokenizer)==151657:
                import json
                with open("ultron/model/assets/special_token.json", "r") as file:
                    special_token = json.load(file)
                self.tokenizer.add_special_tokens({"additional_special_tokens":special_token})
    
    def reset(self):
        self.history = []
          
    def rule_based_instruction(self,env_prompt:str):
        item_name = env_prompt[11:].replace("_"," ")
        instruction_template = random.choice(INSTRUCTION_TEMPLATE)
        return instruction_template.format(item_name)
          
    def create_basic_instruction(self,env_prompt:str):
        instruction = "."
        instructions = self.prompt_library.get(env_prompt,{}).get("instruct")
        if instructions:
            instruction = np.random.choice(instructions)
        else:
            instruction = self.rule_based_instruction(env_prompt)

        if instruction.strip()[-1] != '.':
            instruction += ". \n"
        instruction += "\n"
        return instruction
        
    def get_recipe_item_name(self,ingredient:dict):
        item_name = ingredient.get("item") 
        if not item_name:
            item_name = ingredient.get("tag") 
        return item_name
        
    def create_recipe_prompt_from_library(self,item_name:str):
        if item_name in self.recipes:
            return self.recipes[item_name]
        recipe_path = self.recipe_fold/f"{item_name}.json"
        print(recipe_path)
        if not recipe_path.exists():
            self.recipes[item_name]= ""
            return ""
        recipe_file = load_json_file(recipe_path)
        recipe_type = recipe_file.get("type",None)
        
        prompt = ""
        if not recipe_type:
            self.recipes[item_name]= ""
            return ""
        elif recipe_type=="minecraft:crafting_shapeless":
            prompt+=f"\nYou will need the following ingredients: \n"
            ingredients = recipe_file.get("ingredients",None)
            ingredients_list = []
            for ingredient in ingredients:
                ingredient_name = self.get_recipe_item_name(ingredient)
                if not ingredient_name:
                    break
                ingredients_list.append(ingredient_name[10:].replace("_"," "))
            ingredients_dict = Counter(ingredients_list)
            for item,number in ingredients_dict.items():
                prompt += f"{number} {item}, "
            prompt += "\n"
        elif recipe_type == "minecraft:crafting_shaped":
            prompt+="\nArrange the materials in the crafting grid according to the following pattern: \n"
            patterns = recipe_file.get("pattern",None)
            if not patterns:
                return ""
            ingredients = recipe_file.get("key",{})
            ingredients_dict = {}
            for ingredient_mark,value in ingredients.items():
                ingredient_name = self.get_recipe_item_name(value)
                ingredients_dict[ingredient_mark] = ingredient_name[10:]
            prompt+="\n"
            for pattern_line in patterns:
                for pattern_mark in pattern_line:
                    ingredients_name = ""
                    if pattern_mark==" ":
                        ingredients_name = "air"
                    else:
                        ingredients_name = ingredients_dict.get(pattern_mark,"air")
                    prompt += f" {ingredients_name} |"
                if prompt[-1]=='|':
                    prompt = prompt[:-1] 
                    prompt += "\n"
            prompt +="\n"
        else:
            self.recipes[item_name]= ""
            return ""
        result_num = recipe_file.get("result",{}).get("count",1)
        prompt += f"and get {result_num} {item_name.replace('_',' ')}. \n"
        self.recipes[item_name]= prompt
        return prompt
        
    def create_recipe_prompt(self,env_prompt:str,method:str="crafting_table"):
        """从原始的一句话转换成prompt """
        prompt = ""
        # 如果使用recipe book to craft
        if "recipe_book" in method:
            prompt += "\nUse the recipe book to craft. \n"
            return prompt
        # else
        recipe = self.prompt_library.get(env_prompt,{}).get("recipe")
        if recipe:
            prompt += "\nArrange the materials in the crafting grid according to the following pattern: \n"
            prompt += recipe[0]
        else:
            item_name = env_prompt.replace(" ","_").split(":")[-1]
            prompt += self.create_recipe_prompt_from_library(item_name,)
        return prompt
        
    def create_instruction(self,env_prompt,method):
        prompt =None
        if self.instruction_type == 'recipe':
            prompt = self.create_basic_instruction(env_prompt)
            recipe_prompt = self.create_recipe_prompt(env_prompt,method=method)

            prompt += recipe_prompt
        elif self.instruction_type == 'simple':
            prompt = self.create_thought(env_prompt) 
        elif self.instruction_type == 'normal':
            natural_text = env_prompt.replace("_"," ").replace(":"," ")
            prompt = random.choice(self.prompt_library.get(env_prompt,{"instruct":[natural_text]})["instruct"])
        else:
            raise ValueError(f"do not set the instruction class {self.instruction_type}")
        return prompt
        
    def create_thought(self,env_prompt):
        thought = copy.copy(self.prompt_library.get(env_prompt,{}).get("thought"))
        if not thought:
            thought = env_prompt.replace("item",str(1)).replace("_"," ").replace(":"," ")   #craft item xxx =》 craft 1 item
        thought += '. \n'
        return thought
        
    def forward(self,observations,instructions,verbos=False,need_crafting_table=False):
        if self.actions:
            if verbos:
                print(self.actions)
            if len(self.actions)>1:
                return self.actions.pop(0)
            else:
                action = self.actions[0]
                self.actions = []
                return action
        messages = []
        image = self.processor_wrapper.create_image_input(observations[0]) 
        method = self.method_map[need_crafting_table]
        prompts = []
        private_instruction = self.create_instruction(instructions[0],method=method)
        #print(private_instruction)
        thought= self.create_thought(instructions[0]) if self.instruction_type =="recipe" else ""

        if self.history_num:
            if not self.history: #如果历史为空
                self.history = [(image,self.action_tokenizer.null_token(),copy.copy(thought),0)]*self.history_num
            new_history = [None]*self.history_num
            new_history[:-1] = self.history[1:]
            for hdx,(im, ac, past_thought,_) in enumerate(self.history):
                prompt_input = ""
                if self.instruction_type == 'recipe':
                    prompt_input = "\nthought: " + past_thought + "\nobservation: "  #往上一个prompt上加上这一步的thought
                else:
                    prompt_input = "\nobservation: "
                if not hdx: #hdx==0
                    prompt_input = private_instruction + prompt_input
                #print(ac,prompt_input,)
                messages.append(self.processor_wrapper.create_message_vllm(role="user",input_type="image",prompt=[prompt_input],image=[im]))
                messages.append(self.processor_wrapper.create_message_vllm(role="assistant",input_type="text",prompt=[ac],))
            
        prompt_input = ""
        if self.instruction_type == 'recipe':
            prompt_input = "\nthought: " + thought + "\nobservation: "
        else:
            prompt_input = "\nobservation: "
        if not self.history_num:
            prompt_input = private_instruction + prompt_input
        #print(prompt_input)

        messages.append(self.processor_wrapper.create_message_vllm(role="user",input_type="image",prompt=[prompt_input],image=[image]))
        
        open_logprobs = False
        if verbos:
            print(prompts)
            open_logprobs = True
        

        
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=1024,
            logprobs = open_logprobs,
            extra_body = {"skip_special_tokens":False}
        )

        outputs = chat_completion.choices[0].message.content
        if self.history_num:
            new_history[-1] = (image,outputs,thought,self.history[-1][-1]+1)
            self.history = new_history
        if self.LLM_backbone in {"qwen2_vl"}:
            outputs = self.tokenizer(outputs)["input_ids"]

        actions =  self.action_tokenizer.decode(outputs)

        len_action = min(self.action_chunk_len,len(actions))
        self.actions = actions[:len_action]
        
        return self.actions.pop(0)
