from transformers import AutoModel


model = AutoModel.from_pretrained("CraftJarvis/JarvisVLA-Qwen2-VL-7B")
print("Model loaded successfully.")