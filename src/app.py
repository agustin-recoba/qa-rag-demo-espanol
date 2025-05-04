from rag.data import get_wikipedia_documents
from rag.chunking import chunk_documents
from rag.rag import BiEncoderChunkRetriever, RandomChunkRetriever
from rag.generation import ModelGenerator

from sentence_transformers import SentenceTransformer
import streamlit as st

MODEL_OPTIONS = {
    "KaLM-embedding-multilingual-mini-v1": "HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "multilingual-e5-large-instruct": "intfloat/multilingual-e5-large-instruct",
}

LLM_OPTIONS = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}

st.set_page_config(page_title="RAG Uruguay Demo", layout="wide")
st.title("RAG sobre Historia y Cultura de Uruguay")


@st.cache_resource(show_spinner=True)
def get_chunks():
    docs = get_wikipedia_documents()
    return chunk_documents(docs)


@st.cache_resource(show_spinner=True)
def get_biencoder(model_name):
    return SentenceTransformer(model_name)


chunks = get_chunks()
retriever = None

retrieval_mode = st.sidebar.selectbox(
    "Modo de recuperación", ["Bi-Encoder", "Aleatorio"], index=0
)

if retrieval_mode == "Bi-Encoder":
    model_choice = st.sidebar.selectbox(
        "Modelo Bi-Encoder",
        list(MODEL_OPTIONS.keys()),
        index=0,
        help="Selecciona el modelo de embeddings para recuperación semántica.",
    )
    model = get_biencoder(MODEL_OPTIONS[model_choice])
    retriever = BiEncoderChunkRetriever(chunks, model)
else:
    retriever = RandomChunkRetriever(chunks)

generator = ModelGenerator(retriever)

llm_choice = st.sidebar.selectbox(
    "Modelo Generador (LLM)",
    list(LLM_OPTIONS.keys()),
    index=0,
    help="Selecciona el modelo de lenguaje para la generación de respuestas.",
)

mostrar_chunks = st.sidebar.checkbox(
    "Mostrar Chunks",
    value=False,
    help="Si está marcado, se mostrarán los chunks recuperados.",
)

st.markdown("""
Ingrese una pregunta sobre historia, cultura o demografía de Uruguay. 
El sistema buscará los fragmentos más relevantes y mostrará la respuesta basada en ellos.
""")

question = st.text_input("Pregunta", "¿Quién fue José Gervasio Artigas?")
num_chunks = st.slider("N° de chunks a recuperar", 1, 5, 3)

if st.button("Consultar"):
    with st.spinner("Buscando respuesta..."):
        model_llm, tokenizer_llm = ModelGenerator.load_llm_and_tokenizer(
            LLM_OPTIONS[llm_choice]
        )
        # Recuperar los chunks relevantes primero
        retrieved = retriever.get_n_closest_chunks(question, num_chunks)
        # Generar el prompt usando los chunks recuperados
        prompt = generator.build_prompt(tokenizer_llm, question, retrieved)

        if mostrar_chunks:
            st.markdown(
                "Los siguientes fragmentos fueron recuperados para generar la respuesta:"
            )
            for i, chunk in enumerate(retrieved, 1):
                st.markdown(f"**Chunk {i}:** {chunk}")

        response = generator.generate(model_llm, tokenizer_llm, prompt)
        st.subheader("Respuesta generada:")
        st.write(response)
