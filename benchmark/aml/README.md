# Agent Memory Leaderboard adapter

`server.py` exposes the AML Add/Search HTTP API and translates requests to an
existing OpenViking server. Start OpenViking separately. In a formal AML run,
the platform supplies the datasets and controls Answer, Judge, and score
aggregation.

Run the commands below from the repository root with the project's Python
dependencies installed. The standard OpenViking Docker image and installed
Python package do not include `benchmark/aml`; run the adapter separately from
a checkout containing this directory.

## Add/Search adapter


For a local OpenViking server using `server.auth_mode: dev`:

```bash
export OPENVIKING_URL=http://127.0.0.1:1933
export AML_API_KEY='<memory-system key>'

python -m benchmark.aml.server --host 127.0.0.1 --port 8088
```

Do not set an OpenViking key or account for this local `dev` setup. The SDK
passes each AML `user_id` directly as the OpenViking user; there is no hash or
second identity mapping. Like other OpenViking SDK clients, it also reads an
existing `~/.openviking/ovcli.conf`, so that file must match the server being
used (or be disabled with `OPENVIKING_CLI_CONFIG_FILE` pointing to a missing
file).

For a remote multi-user deployment, configure the OpenViking server with
`server.auth_mode: trusted`, then set the trusted-mode root key and account:

```bash
export OPENVIKING_URL='https://openviking.example.com'
export OPENVIKING_API_KEY='<trusted-mode root key>'
export OPENVIKING_ACCOUNT=default
export AML_API_KEY='<memory-system key>'

python -m benchmark.aml.server --host 0.0.0.0 --port 8088
```

OpenViking's standard `api_key` mode cannot use one root key to switch among
dynamic AML users: in that mode a request must use the key belonging to the
target user. `AML_API_KEY` is unrelated; it only protects this adapter's
incoming `/add` and `/search` endpoints.

The adapter provides:

- `GET /health`
- `POST /add`
- `POST /search`

`/add` calls the OpenViking SDK's `batch_add_messages`, `commit_session`, and
`get_task` in that order. `/search` calls `find` under the same AML user. Each
dataset case is isolated by its `user_id`; session IDs only separate histories
inside that user's space. Transient SDK calls are retried up to three times by
default; use `AML_RETRY_ATTEMPTS` and `AML_RETRY_DELAY_SECONDS` to adjust that
behavior. `Token`, `Bearer`, and `X-Api-Key` authentication are accepted when
`AML_API_KEY` is set.

## Static checks

```bash
python -m ruff check benchmark/aml/server.py benchmark/aml/__init__.py
python -m ruff format --check benchmark/aml/server.py benchmark/aml/__init__.py
```
