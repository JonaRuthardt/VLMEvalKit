import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

class SteerViT(BaseModel):
    def __init__(self, model_path):
        super().__init__()
        assert model_path is not None, "model_path must be provided"
        try:
            from steervit import SteerViT as SteerViTModel
        except ImportError:
            raise ImportError(
                "SteerViT model requires the 'steervit' package. "
                "Please install it to use this model."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SteerViTModel.from_pretrained(model_path).to(self.device).eval()
        self.transform = self.model.get_transforms()

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        answer_subset = None
        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            answer_subset = ["yes", "no"]
        with torch.inference_mode():
            answer = self.model.get_vqa_answer(
                image, [prompt], answer_subset=answer_subset
            )[0]
        return answer