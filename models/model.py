import torch
import torchvision.models as vis_models
import yaml
from models.swin import SwinTransformer  # Replace with actual SwinTransformer import

def build_model(config):
    model = SwinTransformer(**config["base_config"])
    model_w = vis_models.swin_s(weights=vis_models.Swin_S_Weights.IMAGENET1K_V1)
    model.load_state_dict(model_w.state_dict())
    
    if "lora_config" in config:
        model.lorify(
            config["lora_config"]["ranks"],
            config["lora_config"]["alphas"],
            config["lora_config"]["additional_ranks"],
            config["lora_config"]["additional_alphas"]
        )

    return model

def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    yaml_file = "configs/swin_small_lora3.yaml"
    config = load_config(yaml_file)
    model = build_model(config)
    
    print("Model built successfully!")
