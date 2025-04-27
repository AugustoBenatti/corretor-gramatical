# Instala as bibliotecas necessárias
!pip install -q gradio transformers torch

# Importa as bibliotecas usadas
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import time

# Função para carregar o modelo de correção gramatical
def load_model():
    print("Carregando modelo... (isso pode demorar alguns minutos na primeira execução)")
    tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
    model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
    print("Modelo carregado com sucesso!")
    return tokenizer, model

# Chama a função para carregar o modelo
tokenizer, model = load_model()

# Função que faz a correção gramatical do texto
def corrigir_texto(texto):
    if not texto.strip():
        return ""
    
    try:
        start_time = time.time()
        prompt = "gec: " + texto
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
        corrigido = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrigido
    except Exception as e:
        return f" Erro na correção: {str(e)}"

# Cria a interface visual usando Gradio
with gr.Blocks(theme=gr.themes.Soft(), title="Corretor Gramatical no Colab") as interface:
    gr.Markdown("""...""")
    ...
