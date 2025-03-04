import torch
import torch.nn as nn
import torchvision.models as vis_models
import yaml
import numpy as np
from models.BiLSTM import BiLSTMLayer
from models.tconv import TemporalConv
from models.decode import Decode


class SLRModel(nn.Module):
    def __init__(
        self, num_classes, conv_type, use_bn=False,
        hidden_size=1024, gloss_dict=None, share_classifier=True,
        config=None, logger=None
    ):

        super(SLRModel, self).__init__()
        self.visual_dict = {
            "swin_tiny": vis_models.swin_t,
            "swin_small": vis_models.swin_s,
            "resnet18": vis_models.resnet18
        }
        self.visual_weights = {
            "swin_tiny": vis_models.Swin_T_Weights.IMAGENET1K_V1,
            "swin_small": vis_models.Swin_S_Weights.IMAGENET1K_V1,
            "resnet18": vis_models.ResNet18_Weights.IMAGENET1K_V1
        }

        model_name = config["model_args"]["model"]
        self.visual = self.visual_dict[model_name](weights=self.visual_weights[model_name])

        if "swin" in model_name:
            self.visual.head = nn.Linear(768, 512)
        else:
            self.visual.fc = nn.Identity()
    
        gloss_dict = np.load(gloss_dict, allow_pickle=True).item()
        self.num_classes = len(gloss_dict) + 1
        self.decoder = Decode(gloss_dict, num_classes, 'beam')

        if logger is not None: logger(f"Number of classes: {self.num_classes}\n")

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

        self.classifier = nn.Linear(hidden_size, self.num_classes)
        self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)

        if share_classifier:
            self.conv1d.fc = self.classifier

        self.register_full_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def visual_features(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.visual(inputs)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x):
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise = self.visual_features(inputs, len_x)
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
