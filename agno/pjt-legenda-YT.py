# ------------------------------
# Imports e Configurações
# ------------------------------
from dotenv import load_dotenv
import os
import shutil
import yt_dlp
import whisper
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, VideoUnavailable, TranscriptsDisabled, NoTranscriptFound
from agno.agent import Agent
from agno.models.openai import OpenAIChat

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini", api_key=API_KEY),
    show_tool_calls=True,
)

# ------------------------------
# Funções auxiliares
# ------------------------------
def checar_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("⚠️ FFmpeg ou FFprobe não encontrados no PATH.")
        return False
    return True

# ------------------------------
# Definição do estado
# ------------------------------
class VideoState(BaseModel):
    url: str
    texto: str = ""   # legenda ou transcrição
    resumo: str = ""  # resumo final

# ------------------------------
# Funções do fluxo
# ------------------------------
def tentar_legenda(state: VideoState):
    print("\n🔎 Tentando pegar legenda do vídeo...")
    try:
        video_id = state.url.split("v=")[-1].split("&")[0]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt', 'en'])
        state.texto = " ".join([t['text'] for t in transcript_list])
        print("✅ Legendas encontradas.")
    except VideoUnavailable:
        print("❌ Vídeo indisponível.")
        state.texto = ""
    except TranscriptsDisabled:
        print("❌ Legendas desativadas pelo autor do vídeo.")
        state.texto = ""
    except NoTranscriptFound:
        print("❌ Nenhuma legenda encontrada para este vídeo.")
        state.texto = ""
    except Exception as e:
        print(f"[Erro tentar_legenda]: {e}")
        state.texto = ""
    return state

def usar_whisper(state: VideoState):
    if state.texto:
        print("ℹ️ Legenda já encontrada, não é necessário usar Whisper.")
        return state

    if not checar_ffmpeg():
        state.texto = "Não foi possível usar Whisper pois FFmpeg/FFprobe não estão no PATH."
        return state

    print("🎙️ Usando Whisper para transcrição do áudio...")
    audio_path = "video_audio.wav"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'video_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    try:
        print("⬇️ Baixando áudio do vídeo...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([state.url])

        print("📝 Transcrevendo áudio com Whisper...")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        state.texto = result["text"]
        print("✅ Transcrição concluída.")
    except Exception as e:
        print(f"[Erro usar_whisper]: {e}")
        state.texto = "Não foi possível obter a transcrição."
    return state

def resumir(state: VideoState):
    if not state.texto:
        state.resumo = "Não há texto para resumir."
        return state

    print("📝 Resumindo o conteúdo...")
    try:
        resumo = agent.run(f"Resuma o seguinte texto:\n\n{state.texto}")
        # Se o retorno tiver atributo 'content', pega só o texto
        if hasattr(resumo, "content"):
            state.resumo = resumo.content
        else:
            state.resumo = resumo
        print("✅ Resumo concluído.")
    except Exception as e:
        print(f"[Erro resumir]: {e}")
        state.resumo = "Erro ao resumir o texto."
    return state

# ------------------------------
# Função principal
# ------------------------------
def executar_fluxo(url_video: str):
    print("\n▶️ Iniciando processamento do vídeo...")
    estado = VideoState(url=url_video)

    # 1️⃣ tentar legenda
    estado = tentar_legenda(estado)

    # 2️⃣ fallback para Whisper
    estado = usar_whisper(estado)

    # 3️⃣ resumir
    estado = resumir(estado)

    # ✅ resultado final
    print("\n✅ Resumo final:\n")
    print(estado.resumo)

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    url_video = input("Digite a URL do vídeo do YouTube: ")
    executar_fluxo(url_video)


