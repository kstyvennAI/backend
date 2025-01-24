from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
import PyPDF2
import os

app = FastAPI()

# Configuração da API Key do OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuração do CORS para permitir conexões do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Substitua por ["https://<seu-frontend>.github.io"] para mais segurança
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def process_slide(file: UploadFile = File(...)):
    try:
        # Verifica se o arquivo enviado é um PDF válido
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="O arquivo enviado não é um PDF válido.")

        # Salva o arquivo temporariamente no sistema
        file_location = f"/tmp/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Verifica se o arquivo foi salvo corretamente
        if not os.path.exists(file_location):
            raise HTTPException(status_code=500, detail="Falha ao salvar o arquivo enviado.")

        # Extrai o texto do PDF
        pdf_text = extract_text_from_pdf(file_location)
        if not pdf_text:
            raise HTTPException(status_code=400, detail="Não foi possível extrair texto do PDF.")

        # Envia o texto para a API GPT-4
        summary = generate_summary_with_gpt4(pdf_text)
        mind_map_html = generate_mind_map_html(summary)

        # Remove o arquivo temporário
        os.remove(file_location)

        # Retorna a resposta para o frontend
        return JSONResponse(content={"summary": summary, "map": mind_map_html})

    except PyPDF2.errors.PdfReadError:
        raise HTTPException(status_code=400, detail="Erro ao ler o arquivo PDF. O arquivo pode estar corrompido.")
    except Exception as e:
        # Log detalhado para depuração
        print(f"Erro ao processar o arquivo: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao processar o arquivo.")

def extract_text_from_pdf(file_path):
    try:
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        if not text:
            raise ValueError("Nenhum texto encontrado no PDF.")
        return text
    except PyPDF2.errors.PdfReadError as e:
        print(f"Erro ao ler o PDF: {e}")
        raise HTTPException(status_code=400, detail="Erro ao ler o arquivo PDF.")
    except Exception as e:
        print(f"Erro inesperado ao processar o PDF: {e}")
        raise HTTPException(status_code=500, detail="Erro ao processar o arquivo.")

def generate_summary_with_gpt4(text):
    try:
        response = openai.Completion.create(
            model="gpt-4",
            prompt=f"Resuma o seguinte conteúdo em um formato didático e organizado:\n\n{text}",
            max_tokens=500,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Erro na API OpenAI: {e}")
        raise HTTPException(status_code=500, detail="Erro ao se comunicar com a API OpenAI.")

def generate_mind_map_html(summary):
    try:
        mind_map = f"""
        graph TD
        A[Resumo Didático] -->|Resumo| B[{summary[:50]}...]
        """
        return mind_map
    except Exception as e:
        print(f"Erro ao gerar o mapa mental: {e}")
        raise HTTPException(status_code=500, detail="Erro ao gerar o mapa mental.")
