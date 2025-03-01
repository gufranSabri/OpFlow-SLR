import torch
import torch.nn as nn
import torchvision.models as vis_models
import yaml
import numpy as np
from models.swin import SwinTransformer  # Replace with actual SwinTransformer import
from models.BiLSTM import BiLSTMLayer
from models.tconv import TemporalConv
from models.decode import Decode


class SLRModel(nn.Module):
    def __init__(
        self, num_classes, conv_type, use_bn=False,
        hidden_size=1024, gloss_dict=None,
        weight_norm=True, share_classifier=True,
        config=None, logger=None
    ):

        super(SLRModel, self).__init__()
        self.swin_dict = {
            "swin_tiny": vis_models.swin_t,
            "swin_small": vis_models.swin_s,
        }
        self.swin_weights = {
            "swin_tiny": vis_models.Swin_T_Weights,
            "swin_small": vis_models.Swin_S_Weights,
        }
        
        self.num_classes = num_classes

        model_w = self.swin_dict[config["model_args"]["model"]](weights=self.swin_weights[config["model_args"]["model"]])
        self.visual = SwinTransformer(**config["swin_base_config"], logger=logger)
        self.visual.load_weights(model_w)

        self.conv2d = vis_models.resnet18(pretrained=True)
        self.conv2d.fc = nn.Identity()

        if "lora_config" in config:
            self.visual.lorify(
                config["lora_config"]["ranks"],
                config["lora_config"]["alphas"],
                config["lora_config"]["additional_ranks"],
                config["lora_config"]["additional_alphas"]
            )

        gloss_dict = np.load(gloss_dict, allow_pickle=True).item()
        self.decoder = Decode(gloss_dict, num_classes, 'beam')

        self.conv1d = TemporalConv(input_size=512,
            hidden_size=hidden_size,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=num_classes
        )
        self.temporal_model = BiLSTMLayer(
            rnn_type='LSTM', 
            input_size=hidden_size, 
            hidden_size=hidden_size,
            num_layers=2, 
            bidirectional=True
        )

        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def forward(self, x, len_x):
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)

        framewise = self.visual(inputs)
        framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)

        conv1d_outputs = self.conv1d(framewise, len_x)
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])

        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }


def build_model(config, logger):
    model = SLRModel(
        num_classes=config["model_args"]["num_classes"],
        conv_type=config["model_args"]["conv_type"],
        use_bn=config["model_args"]["use_bn"],
        hidden_size=config["model_args"]["hidden_size"],
        gloss_dict=config["data"]["gloss_dict_path"],
        weight_norm=config["model_args"]["weight_norm"],
        share_classifier=config["model_args"]["share_classifier"],
        config=config,
        logger=logger
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
