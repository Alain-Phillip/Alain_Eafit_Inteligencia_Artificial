# Respuestas del workshop RAG
Escribe aquí tus respuestas con tus propias palabras. Pega tablas, resultados y enlaces a las capturas de `evidencias/`.

---

# Bloque A · App RAG empresarial

## A1 · Ejecuta la app RAG de punta a punta

_Evidencia:_
- Índice construido: **4 documentos de demostración → 4 chunks**.
- Interfaz Gradio con URL pública temporal.
- Configuración: generación `nvidia/nemotron-3.5-super-vl-120b-a12b`; embeddings `intfloat/multilingual-e5-base` (HuggingFace local); distancia **coseno**; retrieval **k=3, threshold=0.3**.

_Mi respuesta / análisis:_
La app corre de punta a punta: pregunta → recuperación de chunks relevantes → generación fundamentada en el contexto → muestra de fuentes. 
Como una pequeña nota, los modelos de embedding de NVIDIA del notebook original (`nemotron-3-embed-1b`) devolvían **404 / fuera de servicio**, por lo que se cambió a un embedding local de HuggingFace sin tocar el generador.


## A2 · Tu propio corpus de documentos

_Evidencia:_ —

_Mi respuesta / análisis:_
En la ejecución se usó el conjunto `DEMO_DOCUMENTS` (4 docs).


## A3 · Ingeniería del prompt (antes y después)

_Evidencia:_ —

_Mi respuesta / análisis:_
El prompt actual (`system`) es: *"Eres un asistente empresarial. Responde únicamente con información del contexto proporcionado… Responde en español y de forma concisa."*

## A4 · Muestra el score de similitud en las fuentes

_Evidencia:_ —

_Mi respuesta / análisis:_
La app muestra los **títulos** de las fuentes pero no sus **scores**.

---

# Bloque B · RAG sobre papers

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

_Evidencia:_
- 2 papers indexados: **Attention Is All You Need** y **RAG**; **total 150 chunks** (`chunk_size=900`, `chunk_overlap=120`).
- Embeddings `nvidia/llama-nemotron-embed-vl-1b-v2` (dim **2048**); generación `nvidia/nemotron-3.5-super-vl-120b-a12b`; retrieval **k=5, threshold=0.25**.
- Quiz vs *ground truth*:
  - **Q1** (BLEU WMT14 EN-DE): respondió **28.4** citando *Attention Is All You Need* — coincide.
  - **Q2** (memoria paramétrica / no-paramétrica): describe BART (paramétrica) + índice denso de Wikipedia vía DPR (no-paramétrica).
  - **Q3** (factor 1/√dk): explica el control de la magnitud de los productos punto y los gradientes del softmax.
  - **Q4** (RAG-Sequence vs RAG-Token): correcto.
  - **Q5** (8 heads, dk=dv=64): correcto.

_Mi respuesta / análisis:_
Las **5 respuestas factuales fueron correctas** y cada una citó el paper adecuado. El retriever trajo contexto del paper correcto en cada pregunta, lo que indica que el pipeline recupera y fundamenta bien sobre preguntas directas de un solo documento.


## B2 · Experimentos con chunk_size, k y score_threshold

_Evidencia:_ 

_Mi respuesta / análisis:_
Realizando un rápido experimento con un chunk_size de 450 (la mitad) con un chunk_overlap de 60 (también la mitad), no se observaron cambios notables en las respuestas.


## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

_Evidencia:_
Pregunta bonus (conecta ambos papers). Respuesta del RAG: *"The provided excerpts do not contain enough evidence to identify the specific seq2seq model… or to confirm that it is based on the Transformer."* Las 5 fuentes recuperadas fueron de `rag_paper.pdf`.

_Mi respuesta / análisis:_
El modelo no alucinó, admitió explícitamente que le faltaba evidencia, respetando la instrucción del prompt.
Un detalle importante es que la respuesta correcta sí está en el RAG paper, así que el fallo no fue del generador sino del retriever — recuperó 5 chunks del RAG paper pero ninguno que nombrara "BART" ni ningún chunk del paper de Attention. Es decir, "admitió que no sabía" correctamente dado su contexto incompleto, mostrandonos que la calidad de un RAG depende primero de la recuperación. Un buen generador no puede compensar una recuperación que no trae la evidencia.


## B4 · Preguntas en español sobre papers en inglés

_Evidencia:_ —

_Mi respuesta / análisis:_
Repitiendo la pregunta q3 este devolvio la misma respuesta correctamente en español.

---

# Reflexión

## R1 · Reflexión final con tus datos

_Evidencia:_ Ejecución del notebook 03 (quiz + pregunta bonus) y del notebook 02b (app empresarial).

_Mi respuesta / análisis:_

**1. ¿Qué ocurrió cuando el contexto no contenía información suficiente?**
El modelo reconoció la limitación. Ante la pregunta bonus no logró reunir el contexto de ambos papers y, en lugar de inventar, indicó explícitamente que la evidencia disponible no era suficiente, ofreciendo solo lo que sí podía sustentar.

**2. ¿Qué fragmentos recuperó para la pregunta bonus? ¿Aparecieron ambos papers?**
No. El retriever recuperó 5 fragmentos y todos provenían de un solo paper (`rag_paper.pdf`); el paper de *Attention* no apareció.

**3. ¿Qué efecto tendría modificar `chunk_size`, `chunk_overlap`, `k` o `score_threshold`?**
Aumentando chunk_size trabajariamos con bloques de información más grandes (más tokens/caracteres por chunk), al aumentar chunk_overlap tendríamos mayor redundancia entre chunks adyacentes y con ello evitamos que se pierda contexto entre estas. Cambiar k modifica la cantidad de chunks que puede recuperar, limitando la información que puede tener de contexto y score_threshold es que tanta similaritud tienen que tener los chunks con la pregunta para ser aceptados en el retrival de estos.

**4. ¿Por qué conviene separar el modelo de embeddings del generativo?**
Más que todo, la separación está dada por la estructura que estos utilizan. Mientras el encoder recibe texto y entrega vectores, el generativo recibe texto y entrega texto. El objetivo del encoder es análizar que tan cercanos son los textos que se le entregan mientras el del generador es ver "que continua" en la frase que está generando.

**5. ¿Qué cambiarías para llevar este prototipo a producción?**
- **Persistencia:** usar un índice que no dependa de la sesión de Colab.
- **Control de acceso:** autenticación con login, ya que ahora cualquiera con el link puede usarlo.
- **Evaluación:** usar LLMs más grandes o métricas automáticas para evaluar las respuestas de forma automática y repetible.
- **Observabilidad:** almacenar y monitorear métricas del proceso.
- **Costos:** controlar qué modelos se usan y cachear respuestas frecuentes.
