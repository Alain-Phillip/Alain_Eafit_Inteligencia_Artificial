# Respuestas del workshop RAG
Escribe aquí tus respuestas con tus propias palabras. Pega tablas, resultados y enlaces a las capturas de `evidencias/`.

---

# Bloque 0 · Teoría

> Nota: estas respuestas son conceptuales (teoría de HNSW/FAISS). Confirma el enunciado exacto de cada ejercicio en la sección de Teoría del sitio y completa el campo *Evidencia* con la salida de tu ejecución del Bloque 0.

## T1 · Cuántos nodos viven en cada capa

_Evidencia:_ (pendiente: salida del ejercicio que imprime el conteo de nodos por capa)

_Mi respuesta / análisis:_
HNSW es un grafo multicapa. La **capa 0 (base) contiene TODOS los N nodos**. A cada nodo se le asigna una capa máxima con una probabilidad que **decae exponencialmente**, así que las capas superiores tienen cada vez menos nodos: aproximadamente N, N/M, N/M², … hacia arriba, hasta la capa superior con muy pocos (idealmente 1, el punto de entrada). Esa jerarquía —muchos nodos abajo con enlaces cortos, pocos arriba con enlaces largos— es lo que permite navegar en ~O(log N).


## T2 · ef, k y el tamaño de la cola

_Evidencia:_ (pendiente)

_Mi respuesta / análisis:_
`ef` (ef_search) es el **tamaño de la lista dinámica de candidatos** (la cola de prioridad) que HNSW mantiene durante la búsqueda. Debe cumplirse **ef ≥ k**. `k` es cuántos resultados finales devuelves; `ef` es cuán **ancha** es la exploración mientras los busca. Un `ef` grande examina más candidatos → **mayor recall pero más lento**; si `ef = k`, la búsqueda es más rápida pero con menor recall.


## T3 · Inserción frente a búsqueda

_Evidencia:_ (pendiente)

_Mi respuesta / análisis:_
Insertar un nodo en HNSW es **"buscar + conectar"**: primero se hace una búsqueda (usando `ef_construction`) para hallar los vecinos más cercanos del nuevo nodo, y luego se enlaza con hasta `M` aristas por capa. Por eso **la inserción cuesta más que una búsqueda simple** (recorre Y modifica el grafo), y construir el índice ≈ N inserciones. La búsqueda solo recorre el grafo; la inserción lo recorre y lo escribe.


## T4 · El invariante de FAISSVectorStore

_Evidencia:_ (pendiente)

_Mi respuesta / análisis:_
El invariante es la **correspondencia 1-a-1 entre cada vector del índice y su id/documento externo**: el vector i-ésimo añadido corresponde al id/metadato i-ésimo, y esa alineación debe mantenerse siempre. Además, la **dimensión se fija al crear el índice** y todo vector añadido debe tener esa misma dimensión; el número de vectores almacenados = número de ids en el docstore. Si se rompe esa alineación, las búsquedas devuelven los documentos equivocados.


## T5 · El costo de la fuerza bruta

_Evidencia:_ (pendiente)

_Mi respuesta / análisis:_
La búsqueda exacta (*flat*) calcula la distancia de la query a **cada uno de los N vectores** de dimensión d → **O(N·d)** por consulta. Es 100% exacta (recall perfecto) pero escala **linealmente con N**, así que se vuelve inviable en corpus grandes. Es el *baseline* contra el que se miden los métodos aproximados (HNSW, IVF), que sacrifican un poco de recall a cambio de una búsqueda sublineal.


---

# Bloque A · App RAG empresarial

## A1 · Ejecuta la app RAG de punta a punta

_Evidencia:_
- Índice construido: **4 documentos de demostración → 4 chunks** (`chunk_size=400`, `chunk_overlap=40`).
- Interfaz Gradio publicada con URL pública temporal (`https://…gradio.live`).
- Configuración: generación `nvidia/nemotron-3.5-super-vl-120b-a12b`; embeddings `intfloat/multilingual-e5-base` (HuggingFace local); vector store **Qdrant en memoria**, distancia **coseno**; retrieval **k=3, threshold=0.3**.

_Mi respuesta / análisis:_
La app corre de punta a punta: pregunta → recuperación de chunks relevantes → generación fundamentada en el contexto → muestra de fuentes. Detalle de implementación: los modelos de embedding de NVIDIA del notebook original (`nemotron-3-embed-1b`) devolvían **404 / fuera de servicio**, por lo que se migró a un **embedding local de HuggingFace sin tocar el generador** — una demostración práctica de la modularidad del pipeline (ver R1-4).


## A2 · Tu propio corpus de documentos

_Evidencia:_ —

_Mi respuesta / análisis:_
**PENDIENTE.** En la ejecución se usó el conjunto `DEMO_DOCUMENTS` (4 docs). Para completar A2 hay que crear un `company_documents.json` propio con los campos `id`, `title`, `content` (+ `metadata` opcional) y reconstruir el índice; luego documentar aquí qué corpus se usó y una consulta de ejemplo con su respuesta.


## A3 · Ingeniería del prompt (antes y después)

_Evidencia:_ —

_Mi respuesta / análisis:_
**PENDIENTE.** El prompt actual (`system`) es: *"Eres un asistente empresarial. Responde únicamente con información del contexto proporcionado… Responde en español y de forma concisa."* Para A3 hay que mostrar un **antes/después**: ejecutar una pregunta con el prompt original, luego modificarlo (p. ej. exigir que cite el título de la fuente, o estructurar la respuesta) y comparar ambas salidas.


## A4 · Muestra el score de similitud en las fuentes

_Evidencia:_ —

_Mi respuesta / análisis:_
**PENDIENTE.** La app muestra los **títulos** de las fuentes pero no sus **scores**. Para A4 hay que usar `vectorstore.similarity_search_with_score(...)` (o recuperar los scores del retriever) y mostrar el score de coseno junto a cada fuente en `sources_md`.


---

# Bloque B · RAG sobre papers

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

_Evidencia:_
- 2 papers indexados: **Attention Is All You Need** (40,174 caracteres → 55 chunks) y **RAG** (69,121 caracteres → 95 chunks); **total 150 chunks** (`chunk_size=900`, `chunk_overlap=120`).
- Embeddings `nvidia/llama-nemotron-embed-vl-1b-v2` (dim **2048**); generación `nvidia/nemotron-3.5-super-vl-120b-a12b`; retrieval **k=5, threshold=0.25**.
- Quiz vs *ground truth*:
  - **Q1** (BLEU WMT14 EN-DE): respondió **28.4** citando *Attention Is All You Need* — ✓ coincide.
  - **Q2** (memoria paramétrica / no-paramétrica): ✓ describe BART (paramétrica) + índice denso de Wikipedia vía DPR (no-paramétrica).
  - **Q3** (factor 1/√dk): ✓ explica el control de la magnitud de los productos punto y los gradientes del softmax.
  - **Q4** (RAG-Sequence vs RAG-Token): ✓ correcto.
  - **Q5** (8 heads, dk=dv=64): ✓ correcto.

_Mi respuesta / análisis:_
Las **5 respuestas factuales fueron correctas** y cada una citó el paper adecuado. El retriever trajo contexto del paper correcto en cada pregunta, lo que confirma que el pipeline recupera y fundamenta bien sobre preguntas directas de un solo documento.


## B2 · Experimentos con chunk_size, k y score_threshold

_Evidencia:_ —

_Mi respuesta / análisis:_
**PENDIENTE** (la ejecución usó los valores por defecto 900/120, k=5, threshold=0.25). Para B2 hay que **variar** estos parámetros y comparar recuperación/calidad — por ejemplo, subir `k` para la pregunta bonus y ver si recupera ambos papers. El efecto conceptual está razonado en R1-pregunta 3.


## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

_Evidencia:_
Pregunta bonus (conecta ambos papers). Respuesta del RAG: *"The provided excerpts do not contain enough evidence to identify the specific seq2seq model… or to confirm that it is based on the Transformer."* Las **5 fuentes recuperadas fueron todas de `rag_paper.pdf`**.

_Mi respuesta / análisis:_
El modelo **no alucinó**: admitió explícitamente que le faltaba evidencia, respetando la instrucción del prompt. Matiz importante: la respuesta correcta (BART, basado en Transformer) **sí está** en el RAG paper, así que el fallo no fue del generador sino del **retriever** — recuperó 5 chunks del RAG paper pero ninguno que nombrara "BART" ni ningún chunk del paper de Attention. Es decir, "admitió que no sabía" **correctamente dado su contexto incompleto**. Esto ilustra que la calidad de un RAG depende primero de la recuperación: un buen generador no puede compensar una recuperación que no trae la evidencia.


## B4 · Preguntas en español sobre papers en inglés

_Evidencia:_ —

_Mi respuesta / análisis:_
**PENDIENTE.** El prompt pide *"Answer in the same language as the question"*, pero en la ejecución todas las preguntas del quiz fueron en inglés. Para B4 hay que hacer una pregunta **en español** sobre contenido en inglés y verificar que responde en español fundamentándose en el contexto inglés recuperado.


## B5 · Escribe tu propio quiz con ground truth

_Evidencia:_ —

_Mi respuesta / análisis:_
**PENDIENTE.** Hay que redactar preguntas propias con su *ground truth* (sobre cualquiera de los dos papers) y evaluar las respuestas del RAG contra ellas, anotando aciertos y de qué paper provino el contexto.


---

# Reflexión

## R1 · Reflexión final con tus datos

_Evidencia:_ Ejecución del notebook 03 (quiz + pregunta bonus) y del notebook 02b (app empresarial).

_Mi respuesta / análisis:_

**1. ¿Qué ocurrió cuando el contexto no contenía información suficiente?**
El modelo reconoció la limitación: ante la pregunta bonus no logró reunir el contexto de ambos papers y, en lugar de inventar, indicó explícitamente que la evidencia disponible no era suficiente, ofreciendo solo lo que sí podía sustentar.

**2. ¿Qué fragmentos recuperó para la pregunta bonus? ¿Aparecieron ambos papers?**
No. El retriever recuperó 5 fragmentos y **todos provenían de un solo paper** (`rag_paper.pdf`); el paper de *Attention* no apareció.

**3. ¿Qué efecto tendría modificar `chunk_size`, `chunk_overlap`, `k` o `score_threshold`?**
Aumentar `chunk_size` → bloques de información más grandes (más tokens/caracteres por chunk). Aumentar `chunk_overlap` → más redundancia entre chunks adyacentes, evitando perder contexto en los bordes. Cambiar `k` → modifica cuántos chunks se recuperan, limitando o ampliando la información de contexto. `score_threshold` → fija qué tan similares deben ser los chunks a la pregunta para ser aceptados en la recuperación.

**4. ¿Por qué conviene separar el modelo de embeddings del generativo?**
La separación responde a que tienen **estructuras y objetivos distintos**: el encoder recibe texto y entrega vectores (su objetivo es medir qué tan cercanos son los textos), mientras el generativo recibe texto y entrega texto (su objetivo es predecir "qué continúa"). Además, esa **modularidad** permite cambiar uno sin tocar el otro — como ocurrió al migrar los embeddings a HuggingFace sin modificar el generador.

**5. ¿Qué cambiarías para llevar este prototipo a producción?**
- **Persistencia:** usar un índice que no dependa de la sesión de Colab (Qdrant persistente / en disco).
- **Control de acceso:** autenticación con login, ya que ahora cualquiera con el link puede usarlo (y gastar créditos).
- **Evaluación:** usar LLMs más grandes o métricas automáticas (tipo RAGAS) para evaluar las respuestas de forma automática y repetible.
- **Observabilidad:** almacenar y monitorear métricas del proceso (latencia, chunks recuperados con sus scores, tokens y costo por consulta).
- **Costos:** controlar qué modelos se usan (modelo barato para preguntas fáciles) y cachear respuestas frecuentes.
