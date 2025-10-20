import os
import warnings
import torch
from PIL import Image

from .llava import LLaVA

# def extract_option_choice(text, options_str):
#     import re
#     """응답에서 선택지(A/B)를 추출합니다."""
#     text = text.lower()
    
#     # 직접적인 옵션 문자 확인 - 새 프롬프트 형식에 맞춤
#     if re.search(r'\ba\b', text) and not re.search(r'\bb\b', text):
#         return "(a)"
#     elif re.search(r'\bb\b', text) and not re.search(r'\ba\b', text):
#         return "(b)"
    
#     # 첫 문장에서 옵션 문자 확인
#     first_sentence = text.split('.')[0].lower()
#     if 'a' in first_sentence and 'b' not in first_sentence:
#         return "(a)"
#     if 'b' in first_sentence and 'a' not in first_sentence:
#         return "(b)"
    
#     # 선택지 파싱 - 원래 형식과 새 형식 모두 처리
#     option_a_content = ""
#     option_b_content = ""
    
#     # 원래 형식 (a) ... (b) ...
#     options_match = re.search(r'\(a\)\s*([^()]+)\s*\(b\)\s*([^()]+)', options_str.lower())
#     if options_match:
#         option_a_content = options_match.group(1).strip()
#         option_b_content = options_match.group(2).strip()
    
#     # 새 형식 A. ... B. ...
#     else:
#         options_match = re.search(r'A\.\s*([^AB]+)\s*B\.\s*([^AB]+)', options_str)
#         if options_match:
#             option_a_content = options_match.group(1).strip()
#             option_b_content = options_match.group(2).strip()
    
#     # 선택지 내용 기반 매칭
#     if option_a_content and option_b_content:
#         if option_a_content in text:
#             # option_a가 포함되어 있고 option_b가 포함되어 있지 않거나,
#             # option_a가 더 빨리 언급됨
#             if option_b_content not in text or text.find(option_a_content) < text.find(option_b_content):
#                 return "(a)"
        
#         if option_b_content in text:
#             # option_b가 포함되어 있고 option_a가 포함되어 있지 않거나,
#             # option_b가 더 빨리 언급됨
#             if option_a_content not in text or text.find(option_b_content) < text.find(option_a_content):
#                 return "(b)"
    
#     # 키워드 기반 추출 - 확장
#     if any(word in text for word in ["first", "former", "a)", "a)", "a.", "option a", "choice a", "a"]):
#         return "(a)"
#     if any(word in text for word in ["second", "latter", "b)", "b)", "b.", "option b", "choice b", "b"]):
#         return "(b)"
    
#     # 빈도수 기반 확인 (마지막 수단)
#     a_count = len(re.findall(r'\ba\b', text))
#     b_count = len(re.findall(r'\bb\b', text))
    
#     if a_count > b_count:
#         return "a"
#     elif b_count > a_count:
#         return "b"
    
#     # 기본값
#     print(f"Warning: Could not extract choice from: '{text}' for options: {options_str}")
#     return "unknown"

class VIRAL(LLaVA):
    def __init__(self,
                 model_path,
                 **kwargs):
        assert model_path is not None and os.path.exists(model_path)
        # super().__init__(model_path=model_path, **kwargs)
        
        try: 
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        except ImportError:
            raise ImportError('Please install llava/viral package correctly. ')

        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = "</s>"
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base="lmsys/vicuna-7b-v1.5" if "7b" in model_path else "lmsys/vicuna-13b-v1.5",
            model_name="llava-v1.5-7b-lora" if "7b" in model_path else "llava-v1.5-13b-lora",
            device_map="cpu",
        )
        self.conv_mode = "llava_v1"
        self.model.vra_loss = False
        self.model = self.model.cuda()
        
        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )
        
    def generate_inner(self, message, dataset=None):
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from abc import abstractproperty

        # Support interleave text and image
        content, images = self.concat_tilist(message)

        images = [Image.open(s).convert("RGB") for s in images]
        if images:
            image_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().cuda()
            # args = abstractproperty()
            # args.image_aspect_ratio = "pad"
            # image_tensor = process_images(images, self.image_processor, args).to(
            #     "cuda", dtype=torch.float16
            # )
        else:
            image_tensor = None

        prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "

        # prompt = prompt.replace("Please answer yes or no.", "Answer with only 'yes' or 'no'.") #POPE
        # prompt = prompt.replace("Base your answer on reasoning. Your final answer must be only the single capital letter corresponding to the correct choice.", "Answer with the option's letter from the given choices directly.") #MMStar
        # prompt = prompt.replace("Answer with the option's letter from the given choices directly.", "Base your answer on reasoning, but answer with the option's letter from the given choices directly.") #MMVP
        
        # self.kwargs["do_sample"] = True
        # self.kwargs["temperature"] = 0.1
        # self.kwargs["top_p"] = 0.7
        # print(prompt)
        # exit()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                stopping_criteria=[stopping_criteria],
                **self.kwargs,
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # content = content.split("?")[-1]
        # content = content.split("\n")[:-1]
        # content = "\n".join(content).strip()
        # output = extract_option_choice(output, content)
        
        # if output.lower() in ["yes", "yeah", "correct", "right", "true", "indeed", "affirmative"]:
        #     output = "yes"
        
        # import re
        # text = output.lower().replace("(", "").replace(")", "").strip()
        # if re.search(r'\b(a|option a)\b', text):
        #     return "A"
        # elif re.search(r'\b(b|option b)\b', text):
        #     return "B"
        # elif re.search(r'\b(c|option c)\b', text):
        #     return "C"
        # elif re.search(r'\b(d|option d)\b', text):
        #     return "D"
        
        return output