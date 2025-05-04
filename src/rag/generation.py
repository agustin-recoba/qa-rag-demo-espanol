"""
Generación de respuestas con LLM y construcción de prompts para RAG.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
import torch


class ModelGenerator:
    def __init__(self, chunk_retriever):
        self.chunk_retriever = chunk_retriever

    @staticmethod
    def load_llm_and_tokenizer(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        return model, tokenizer

    def build_prompt(
        self, tokenizer, question, retrieved_chunks, few_shots_examples=[]
    ):
        messages = [
            {
                "role": "system",
                "content": "Eres un sistema de preguntas y respuestas. Da respuestas CONCRETAS y PRECISAS. Utiliza ÚNICAMENTE la información de los siguientes extractos para construir la respuesta:",
            },
        ]
        for chunk in retrieved_chunks:
            messages.append({"role": "system", "content": f"extract: {chunk}"})
        messages.append(
            {
                "role": "system",
                "content": "ES MUY IMPORTANTE que si los extractos no tienen una relación evidente con la pregunta, debes responder que no se cuenta con información para responder la pregunta.",
            }
        )
        for q, r in few_shots_examples:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": r})
        messages.append({"role": "user", "content": question})
        return tokenizer.apply_chat_template(messages, tokenize=False)

    def generate(self, model, tokenizer, prompt, temp=0.0, max_tok=300):
        generation_config = GenerationConfig(
            temperature=temp if temp > 0 else None, do_sample=temp > 0
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            config=generation_config,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
        )
        output = pipe(prompt, return_full_text=False, max_new_tokens=max_tok)
        return output[0]["generated_text"]
