# autosar-chat
This is a AUTOSAR documents specific retriever based on LLM and RAG.

## Environmental requirements

- Operation System
    - Linux/MacOS (recommend)
    - Windwons
- Docker
- Python: 3.9.6 (compatibility issue may occur if version too hign)

## Installation

### 1. [Install](https://github.com/ollama/ollama?tab=readme-ov-file#macos) ollama and pull models

Pull the `llama3`:

```shell
ollama pull llama3
```

Pull the Embeddings model:

```shell
ollama pull nomic-embed-text
```

### 2. Create a virtual environment

```shell
python3 -m venv ~/.venvs/aienv
source ~/.venvs/aienv/bin/activate
```

### 3. Install libraries

```shell
pip install -r cookbook/requirements.txt
```

### 4. Install Docker

> Install docker according to this doc [Docker](https://docs.docker.com/get-started/get-docker/)

### 5. Run PgVector

- Run using a helper script

```shell
./cookbook/run_pgvector.sh
```

- OR run using the docker run command

```shell
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  phidata/pgvector:16
```
### 6. Run Qdrant 

- Run using a helper script

```shell
./cookbook/run_qdrant.sh
```
- OR run using the docker run command

```shell
docker run -itd --name=qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -e "QDRANT__SERVICE__API_KEY=123456" \
    -e "QDRANT__SERVICE__JWT_RBAC=true" \
    -v /home/qdrant_storage:/qdrant/storage:z \
        qdrant/qdrant
```

NOTE:

If following error occured in the step 5 or step 6, please try to change the mirror of docker

> docker: error pulling image configuration: download failed after attempts=6: dial tcp 199.16.156.11:443: connect: connection refused.


### 7. Run RAG App

```shell
streamlit run app.py
```

- Open [localhost:8501](http://localhost:8501) to view your local RAG app.

- Add PDFs or ask question directly
- Example PDF: ./cookbook/data/AUTOSAR/AUTOSAR_CP_SWS_DiagnosticCommunicationManager-54-78.pdf 
