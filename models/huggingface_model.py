import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "facebook/nllb-200-1.3B"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        self.lang_code_map = {
            ("en", "zh"): ("eng_Latn", "zho_Hans"),
            ("zh", "en"): ("zho_Hans", "eng_Latn"),
        }

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        if (src_lang, tgt_lang) not in self.lang_code_map:
            raise ValueError(f"Unsupported language pair: {src_lang}-{tgt_lang}")

        src_code, tgt_code = self.lang_code_map[(src_lang, tgt_lang)]
        self.tokenizer.src_lang = src_code

        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=512,
                num_beams=5,
                early_stopping=True
            )

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
