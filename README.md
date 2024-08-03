# autosar-chat
This is a AUTOSAR documents specific retriever based on LLM and RAG.

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

### 4. Run PgVector

> Install [docker desktop](https://docs.docker.com/desktop/install/mac-install/) first.

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
Note:

If following error occured, please try to change the mirror of docker

> docker: error pulling image configuration: download failed after attempts=6: dial tcp 199.16.156.11:443: connect: connection refused.


### 5. Run RAG App

```shell
streamlit run app.py
```

- Open [localhost:8501](http://localhost:8501) to view your local RAG app.

- Add websites or PDFs and ask question.
- Example PDF: https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf
- Example Websites:
  - https://techcrunch.com/2024/04/18/meta-releases-llama-3-claims-its-among-the-best-open-models-available/?guccounter=1
  - https://www.theverge.com/2024/4/23/24137534/microsoft-phi-3-launch-small-ai-language-model
