# Multi-Agent Travel Planner

Sistema multi-agente per la pianificazione di viaggi basato su [LangChain](https://www.langchain.com/) e [Ollama](https://ollama.com/), con due modalità di esecuzione: con e senza [MCP (Model Context Protocol)](https://modelcontextprotocol.io/).

## Prerequisiti

- Python 3.10+
- Un'istanza Ollama configurata e raggiungibile
- Un file `.env` con le variabili d'ambiente necessarie (es. `OLLAMA_API_KEY`)

## Setup

### 1. Creare il virtual environment

```bash
python -m venv .venv
```

### 2. Attivare il virtual environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```powershell
.venv\Scripts\activate
```

### 3. Installare le dipendenze

```bash
pip install -r requirements.txt
```

## Esecuzione

### Caso senza MCP

Gli agenti specializzati (logistics e recommendations) vengono creati e invocati direttamente all'interno dello stesso processo.

```bash
python no_mcp/main.py
```

### Caso con MCP

Gli agenti specializzati vengono esposti come server MCP separati e il coordinatore li invoca tramite il protocollo MCP via `stdio`.

```bash
python mcp/main.py
```

### Caso con MCP con CodeMode

Gli agenti specializzati vengono esposti come server MCP separati e il coordinatore li invoca tramite il protocollo MCP via `stdio`.
A differenza del precedente scenario, in questo viene utilizzato il [CodeMode](https://gofastmcp.com/servers/transforms/code-mode)

```bash
python mcp_codemode/main.py
```