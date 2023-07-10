import torch
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
from octobert import build_model
from transformers import RobertaTokenizerFast
from utils.inference_utils import img_transform, post_processor, plot_results

dependencies = ["timm", "tqdm", "transformers", "einops"]

def flickr_base_model():
    """
    Our base model initialized from ResNet 50 and RoBERTa-base, pre-trained on Flickr-30k entities.
    """
    model_checkpoint = torch.hub.load_state_dict_from_url(
        url="https://huggingface.co/sled-umich/OctoBERT-flickr/resolve/main/plain_model.pth",
        map_location="cpu",
        check_hash=True)
    model_checkpoint['args'].device = 'cpu'
    tokenizer = RobertaTokenizerFast.from_pretrained(model_checkpoint['args'].text_encoder_type, return_special_tokens_mask=True)
    model = build_model(model_checkpoint['args'])[0]
    model.load_state_dict(model_checkpoint['model'])
    return model, img_transform, tokenizer, post_processor, plot_results
