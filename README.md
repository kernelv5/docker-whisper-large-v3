# Whisper Large V3 Transcription API

Docker Compose service that runs OpenAI Whisper large-v3. Accepts any FFmpeg-supported audio/video file and returns cleaned transcript (text, language, segments). Optional output filter and cache file generation.

## Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- GPU

## Run

```bash
docker compose up -d
```

First run downloads the model (~3GB); subsequent runs use the cached volume. Cache files are stored in `./cache` on the host.

## API

| Endpoint | Description |
|----------|-------------|
| **POST /transcribe** | Upload a file (any FFmpeg-supported format). Returns transcript. Use `?output=...` to filter response. |
| **GET /cache-files** | JSON list of cache files (temp_cache_file_id, date_and_time, file_name). |
| **GET /cache-files/list** | HTML page listing cache files in a table. |
| **GET /health** | Health check. |
| **GET /** | Redirects to Swagger UI. |
| **GET /docs** | Swagger UI (interactive API docs). |
| **GET /redoc** | ReDoc API docs. |

### Output filter (POST /transcribe)

Query parameter `output`: comma-separated keys. Response includes only these keys.

| Value | Response |
|-------|----------|
| `full` (default) | `text`, `language`, `segments` |
| `text` | `{"text": "..."}` |
| `segment` or `segments` | `{"segments": [...]}` |
| `language` | `{"language": "en"}` |
| `gen_cache` | Writes a timestamp-named JSON file to the cache dir and adds `gen_cache: { temp_cache_file_id, date_and_time, file_name }` to the response. |

You can request 2 or more keys, e.g. `output=text,segment` or `output=text,gen_cache`.

## Examples

```bash
# Full response (default)
curl -X POST -F "file=@video.mp4" http://localhost:8030/transcribe

# Text only
curl -X POST -F "file=@video.mp4" "http://localhost:8030/transcribe?output=text"

# Text + segments
curl -X POST -F "file=@video.mp4" "http://localhost:8030/transcribe?output=text,segment"

# Text + gen_cache (writes cache file, returns text and cache metadata)
curl -X POST -F "file=@video.mp4" "http://localhost:8030/transcribe?output=text,gen_cache"

# List cache files (JSON)
curl http://localhost:8030/cache-files
```

- **Port:** 8030  
- **Timeout:** 1 hour (keep-alive)  
- **Cache dir:** `./cache` (mounted at `/app/cache` in container). Set `CACHE_DIR` in the environment to override.  
- **API docs:** Open http://localhost:8030/docs in a browser.
