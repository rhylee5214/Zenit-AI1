# ZENIT API

Base URL: `http://127.0.0.1:8055` by default. Send `Authorization: Bearer zk-...` on every request, except `/v1/health`. In `--local` mode, requests from the same PC skip the key.

Create keys:

```
python -m zenit.api keys create --name rhylee --admin
python -m zenit.api keys create --name friend --rate 20 --daily-tokens 50000
python -m zenit.api keys list
python -m zenit.api keys revoke key_...
```

## Chat

`POST /v1/chat/completions`

Two ways to call it:

* Stateful (ZENIT style): send `message` and an optional `conversation_id`. ZENIT stores the history, applies memory and document search, and returns the conversation id.
* Stateless (OpenAI style): send `messages`. Nothing gets stored.

| Field | Type | Notes |
|---|---|---|
| `model` | string | Defaults to the server default |
| `message` | string | Stateful mode |
| `conversation_id` | string | Continue a conversation |
| `regenerate` | bool | Replace the last assistant reply in `conversation_id` |
| `messages` | list | Stateless mode, roles `system`, `user`, `assistant` |
| `stream` | bool | Server-sent events, OpenAI chunk format |
| `max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`, `stop` | | Sampling |
| `profile` | string | `balanced`, `precise`, `detailed`, `coder`, `creative` |
| `use_memory`, `use_documents` | bool | Default true |
| `document_ids` | list | Limit retrieval to these documents |

Example:

```bash
curl -N http://127.0.0.1:8055/v1/chat/completions \
  -H "Authorization: Bearer $ZENIT_KEY" -H "Content-Type: application/json" \
  -d '{"message": "Explain RoPE in two sentences", "stream": true}'
```

Every response carries a `zenit` object: `generation_id`, `conversation_id`, `model`, `sources` (retrieved passages with labels S1, S2 and so on), `memories` (memories placed in the prompt), `memory_actions` (memories saved or deleted by this message) and `dropped_turns`.

Stop a generation: close the stream, or `POST /v1/generations/{generation_id}/cancel`. The partial reply gets saved with finish reason `cancelled`.

Python with the OpenAI client library, pointed at ZENIT:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8055/v1", api_key="zk-...")
reply = client.chat.completions.create(model="zenit-0.1-nano",
                                       messages=[{"role": "user", "content": "Who are you?"}])
print(reply.choices[0].message.content)
```

The library only formats the request. The answer comes from your ZENIT server.

## Other endpoints

| Method and path | Purpose |
|---|---|
| `GET /v1/health` | Status, version, model count (no key) |
| `GET /v1/me` | Your key and user |
| `GET /v1/models`, `GET /v1/models/{id}` | Models, model cards, behaviour profiles |
| `GET /v1/conversations`, `POST`, `GET/PATCH/DELETE /v1/conversations/{id}` | History |
| `GET/POST/DELETE /v1/memories`, `PATCH/DELETE /v1/memories/{id}` | Memory management |
| `GET/PUT /v1/settings` | `memory_enabled`, `default_profile` |
| `GET /v1/export` | Everything stored about you, as JSON |
| `POST /v1/documents?filename=name.pdf` | Upload (raw body or multipart field `file`) |
| `GET /v1/documents`, `DELETE /v1/documents/{id}`, `POST /v1/documents/search` | Documents |
| `GET /v1/usage` | Requests and tokens per day and model |
| `GET/POST/DELETE /v1/keys` | Key management (admin) |
| `GET /v1/training/runs`, `GET /v1/training/runs/{id}` | Runs and metrics |
| `POST /v1/training/estimate`, `POST /v1/training/runs`, `POST /v1/training/runs/{id}/stop` | Plan, start and stop runs (admin) |
| `GET /v1/registry` | Model registry and roadmap |
| `GET /v1/evals` | Evaluation results |
| `GET /v1/system` | CPU, RAM, disk, GPU |

Errors use `{"error": {"message": ..., "type": ...}}` with status 400, 401, 403, 404, 409 (confirmation needed for a big training job), 413, 415, 422 or 429 (with `Retry-After`).
