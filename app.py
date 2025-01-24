from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import openai
import PyPDF2

app = FastAPI()

# Configure sua API Key do OpenAI
openai.api_key = "sua_api_key"

@app.post("/upload")
async def process_slide(file: UploadFile = File(...)):
    # Salva o arquivo temporariamente
    with open(file.filename, "wb") as f:
        f.write(await file.read())

    # Extrai o texto do PDF
    pdf_text = extract_text_from_pdf(file.filename)

    # Envia o texto para o GPT-4 para resumo e mapa mental
    summary = generate_summary_with_gpt4(pdf_text)
    mind_map_html = generate_mind_map_html(summary)

    return JSONResponse(content={"summary": summary, "map": mind_map_html})


def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text


def generate_summary_with_gpt4(text):
    response = openai.Completion.create(
        model="gpt-4",
        prompt=f"Resuma o seguinte conteúdo em um formato didático e organizado:\n\n{text}",
        max_tokens=500,
    )
    return response.choices[0].text.strip()


def generate_mind_map_html(summary):
    # Converte o resumo para um gráfico Mermaid
    mind_map = f"""
        graph TD
        A[Resumo Didático] -->|Resumo| B[{summary[:50]}...]
    """
    return mind_map
