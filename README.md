# vid2scene

Turn a phone video into an interactive 3D Gaussian-Splat scene. Open-source self-hostable version of [vid2scene.com](https://vid2scene.com).

- **Upload a video** → SfM (GLOMAP / VGGT) → 3D Gaussian Splat training (gsplat) → in-browser viewer
- **Web app + GPU worker + persistent storage** in a single `docker compose` stack
- **Apache-2.0** licensed; runs locally with zero external accounts (HuggingFace token only needed for optional gated models)

---

## Architecture

vid2scene is a small distributed system: a Django web app hosts the vid2scene web platform — the site, accounts, REST API, upload flow, and in-browser 3D viewer — a separate GPU worker runs the heavy 3D-reconstruction pipeline, and the two coordinate through a job queue, a database, and blob storage.

```
   ┌─────────┐
   │ browser │
   └─────────┘
     │      │
HTTP │      │ direct SAS (browser ⇄ blob storage)
     ▼      └──────────────────────────────────┐
   ┌─────────────────────────────┐             │
   │ web — Django + gunicorn      │             │
   │ (port 8000)                  │             │
   └─────────────────────────────┘             │
     │            │            │               │
     │ enqueue    │ read /     │ server-side   │
     │ (RQ)       │ write      │ blob I/O      │
     ▼            ▼            ▼               ▼
   ┌───────┐  ┌──────────┐  ┌──────────────────────────────┐
   │ redis │  │ postgres │  │ azurite (Azure Blob emulator)│
   └───────┘  └──────────┘  └──────────────────────────────┘
     │                            ▲
     │ job                        │ server-side blob I/O
     ▼                            │
   ┌──────────────┐               │
   │ worker (GPU) │───────────────┘
   └──────────────┘
```

| Service | Image | Role |
|---|---|---|
| `web` | [`Web_Dockerfile`](Web_Dockerfile) (`python:3.11-slim` + `gunicorn`, frontend built in a Node stage) | Django app, REST API, and splat viewer. Accepts uploads and **enqueues** processing jobs. |
| `worker` | [`Worker_Dockerfile`](Worker_Dockerfile) (PyTorch + CUDA + COLMAP/glomap/gsplat from source) | Pulls jobs off the queue and runs the ML pipeline: frame extraction, SfM, splat training, LOD/SPZ/SOG export. |
| `redis` | `redis:7` | Two roles: the **[django-rq](https://github.com/rq/django-rq)** job queue — a Redis-backed task queue where `web` pushes jobs and `worker` pops them — and the Django cache. |
| `postgres` | `postgres:16` | Primary database: users, jobs, and scene metadata (titles, camera framing, file paths). |
| `azurite` | `mcr.microsoft.com/azure-storage/azurite` | **[Azurite](https://github.com/Azure/Azurite)**, Microsoft's Azure Blob Storage emulator. **All scenes live here** — uploaded videos and the generated splat files (PLY/SPZ/SOG/LOD) and previews. The browser uploads/downloads them directly via short-lived SAS URLs. |

**How a job flows:** the browser uploads the video straight to blob storage via a SAS URL, then asks `web` to create a job. `web` writes a row to Postgres and pushes a task onto the django-rq queue in Redis. The `worker` pops it, runs the pipeline on the GPU, writes the resulting scene files back to blob storage, and updates the job in Postgres. The browser polls status and finally loads the finished scene straight from blob storage.

The pipeline itself lives in [`vid2scene_core/`](vid2scene_core/) and is orchestrated by the worker. SfM defaults to GLOMAP (no model weights needed). VGGT and SAM3-based panorama background removal are opt-in and require a Hugging Face token (see below). 360° content is supported two ways: stitched equirectangular video (`--equirectangular`), or Insta360 `.insv` recordings processed directly from their dual fisheye streams for better ground/sky detail — see [docs/insv_fisheye.md](docs/insv_fisheye.md).

On the hosted vid2scene deployment each box in the diagram was its own dedicated piece of infrastructure — the web app on one server, one or more GPU workers each on their own server (the django-rq queue lets you scale out by simply adding more workers that pull from it), plus managed Redis, managed PostgreSQL, and real Azure Blob Storage. This self-host stack collapses all of them into a single `docker compose` project on one machine, swapping the managed cloud services for drop-in equivalents (plain Redis/Postgres containers, and **Azurite** standing in for Azure Blob Storage) so it runs with no cloud account.

---

## Setup guide

A start-to-finish walkthrough from a fresh machine to a running stack. If you already have Docker + NVIDIA GPU support, skip to [step 4](#4-get-the-code).

**Before you start — hardware & OS:**
- **Linux** (native) or **Windows 10/11 + WSL2**. macOS can run the web stack but **not** the GPU `worker` (no NVIDIA GPU), so it isn't a target for actually processing videos.
- An **NVIDIA GPU** with a recent driver (the worker compiles COLMAP / gsplat against CUDA).
- **~30 GB free disk** for the worker image + model caches, plus room for your scenes.

### 1. Install Docker

Install **Docker Engine 24+** with the **Compose plugin** (the `docker compose` subcommand). Use Docker's official docs rather than ad-hoc commands:

- **Linux — Docker Engine:** <https://docs.docker.com/engine/install/> (the Compose plugin ships with it; if missing, see <https://docs.docker.com/compose/install/linux/>)
- **Linux — post-install** (run Docker without `sudo`): <https://docs.docker.com/engine/install/linux-postinstall/> — add your user to the `docker` group, then log out/in.
- **Windows / macOS — Docker Desktop:** <https://docs.docker.com/desktop/> (bundles Compose). On Windows, use the **WSL2 backend** and run the project from inside your WSL distro.

Verify:

```bash
docker --version            # 24.x or newer
docker compose version      # v2.x
docker run --rm hello-world # confirms the daemon works
```

### 2. NVIDIA GPU in Docker

The `worker` needs the GPU exposed to containers. You need a working host driver **plus** the NVIDIA Container Toolkit.

**Confirm the host driver works first** (this must already pass):

```bash
nvidia-smi        # should print your GPU(s)
```

**Install the NVIDIA Container Toolkit** via NVIDIA's official, distro-by-distro instructions:

- **Install guide:** <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
- **Configure Docker** (run after install): <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker> — i.e. `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

> **Already have the toolkit?** If `docker info --format '{{.Runtimes}}'` lists `nvidia` — or your `/etc/docker/daemon.json` already has an `nvidia` entry under `"runtimes"` — it's installed and registered, so you can skip straight to verifying.

**Verify Docker can see the GPU:**

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If that prints the GPU table, the worker will too. The compose `worker` service already requests the GPU (`runtime: nvidia` + `deploy.resources.reservations.devices`), so no per-run `--gpus` flag is needed.

**WSL2:** install the NVIDIA driver on **Windows** (the normal Game Ready / Studio driver) — *not* inside the Linux distro. CUDA is passed through to WSL automatically, but you still install `nvidia-container-toolkit` **inside** the WSL distro per the guide above. Make sure `nvidia-smi` works inside WSL before continuing.

### 3. Host networking

Every service in the compose file uses `network_mode: host`. The browser uploads/downloads blobs directly to Azurite at `127.0.0.1:10000`, while web and worker reach postgres/redis/Azurite at `127.0.0.1` server-side — host networking keeps that address identical on every side, so the SAS URLs the server mints work unchanged in the browser.

- **Native Linux Docker** — host networking is built in; nothing to do.
- **Docker Desktop (Windows + WSL2, or macOS)** — host networking is off by default. Enable **Settings → Resources → Network → "Enable host networking"**, then **Apply & restart**.

### 4. Get the code

Clone the repo and initialize the vendored submodules (glomap, gsplat, etc. — required for the worker build):

```bash
git clone https://github.com/samuelm2/vid2scene.git
cd vid2scene
git submodule update --init --recursive
```

### 5. Configure (optional)

All settings have working defaults. Copy the example env only if you want to change something (admin password, contact email, HF token, etc.):

```bash
cp .env.example .env
```

See [Configuration](#configuration) for the available variables.

### 6. Build and run

```bash
docker compose up --build
```

The first build takes a while (the GPU worker image compiles COLMAP / gsplat from source). When it finishes, open <http://localhost:8000> and log in with the default `admin` / `admin`. Upload a video at `/upload/`, wait for the worker to process it, then view the result in the embedded viewer.

The five services — web, worker, redis, postgres, azurite — share host networking so SAS URLs work uniformly for browser, web, and worker. Data persists in named volumes across `up`/`down`; only `docker compose down -v` wipes it. Add `-d` to run detached.

---

## Configuration

Every knob lives in [`.env`](.env.example). All have working defaults; copy and edit what you care about:

```bash
cp .env.example .env
```

| Variable | Default | Notes |
|---|---|---|
| `SECRET_KEY` | dev-only insecure | Fine as-is for local use. **Set your own if you deploy this anywhere others can reach.** |
| `BILLING_ENABLED` | `false` | When off, the Stripe subscription system is hidden and every user gets full access. Turn on (+ set `STRIPE_SECRET_KEY` in the container env) to use the included billing flow. |
| `DJANGO_SUPERUSER_*` | `admin` / `admin` | Bootstrap admin user created on first start. |
| `POSTGRES_*` | `vid2scene` | Database name / user / password for the bundled postgres. |
| `HF_TOKEN` | empty | Only needed for the optional VGGT and SAM3 paths (see below). |
| `RQ_QUEUES` | `internal enterprise high default` | Which RQ queues the worker consumes. |
| `CONTACT_EMAIL` | `contact@example.com` | Email shown in the UI/footer/docs. |
| `SITE_URL` | `http://localhost:8000` | Canonical URL used in API examples, emails, SEO. |
| `UMAMI_WEBSITE_ID` | empty | Optional Umami analytics. Empty = no tracker is rendered. |
| `LANDING_VIDEO_BASE_URL` | empty | Optional CDN base for the landing-page hero video. Empty = poster only. |

---

## Django admin

Django (the web framework powering the backend) comes with a powerful built-in admin website for viewing, configuring, and orchestrating the vid2scene platform. It lives at `/admin/`; log in with the superuser account (`admin` / `admin` unless you changed `DJANGO_SUPERUSER_*`). From there you can search, edit, and delete any record — scenes, users, jobs, and so on.

This project also adds custom tooling on top of it for running the site:

- **Scenes** — curate which ones appear on the examples page and re-run processing on selected jobs.
- **Site alerts** — post site-wide banner notifications.
- **API keys** — issue and inspect per-user API keys.
- **Billing** — review subscriptions, credits, and refunds (only relevant with `BILLING_ENABLED=true`).

There are a lot of cool features. Definitely try it out!

---

## Optional ML model paths (HuggingFace)

The default `reconstruction_method=glomap` uses pure C++ SfM and needs **no model weights, no HF account, no token**. Two paths download gated models from HuggingFace and require some one-time setup:

1. **VGGT** (faster SfM, picked via `reconstruction_method=vggt`) — uses `facebook/VGGT-1B-Commercial`. Note: reconstruction quality is **noticeably worse** than the default GLOMAP, so it's mainly useful for quick tests rather than final scenes.
2. **Equirectangular / 360° background removal** — uses `facebook/sam3`.

To enable either:
1. Create an account at <https://huggingface.co>.
2. Visit each repo's page and click **"agree to license"**.
3. Generate a [user access token](https://huggingface.co/settings/tokens) and put it in `.env`:
   ```
   HF_TOKEN=<your-huggingface-token>
   ```

Downloads are cached in the `hf-cache` named volume, so models persist across `docker compose down`/`up` and only download once.

---

## Development

For frontend or backend work without rebuilding the full Docker stack:

```bash
# Backend (Python)
cd vid2scene_server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ENVIRONMENT=development python manage.py runserver

# Frontend (Svelte/Vite)
cd vid2scene_server/viewer
npm install
npm run dev          # Vite dev server with HMR
# (or `npm run build` to produce the static manifest)
```

In bare-metal dev mode the stack falls back to **sqlite + local redis (127.0.0.1:6379) + local Azurite (127.0.0.1:10000) + console email**. Spin up redis and Azurite separately on those ports.

Run the test suite:
```bash
cd vid2scene_server
python manage.py test
```

The billing-dependent tests pin `BILLING_ENABLED=True` via `@override_settings`, so they pass in either configuration.

---

## Project layout

```
vid2scene_core/        — ML pipeline (frame extraction, SfM, splat training, LOD)
vid2scene_server/      — Django web app (REST API, viewer, admin)
  viewer/              — Svelte + PlayCanvas frontend
  video_processor/     — Job models, RQ tasks, web/dev API
  subscriptions/       — Optional Stripe billing (gated by BILLING_ENABLED)
  user_homebase/       — Auth, dashboards
  examples/            — Examples gallery app + `seed_examples` command
  examples_data/       — Bundled example scenes (scene.spz + preview.jpg + manifest.json)
  svraster_webgl_demo/ — WebGL svraster viewer (/voxel/ route)
Worker_Dockerfile      — GPU worker image (CUDA + COLMAP/glomap/gsplat)
Web_Dockerfile         — Django web image (slim, frontend built in Node stage)
docker-compose.yaml    — Full self-host stack
scripts/               — Entrypoints and ops scripts (web_entrypoint.sh)
```

Submodules (vendored at clone time): glomap, Hierarchical-Localization, spz, sam3, gsplat, vggt, ply-to-sog, 3dgs-autolod, quest-3d-reconstruction, svraster-webgl, simpleomp. See [NOTICE](NOTICE) for full attribution.

---

## Security

This is the **self-host build**, meant to run on a single trusted machine. It runs Django in development mode (`DEBUG` on, `ALLOWED_HOSTS` limited to localhost) and ships with insecure defaults — `admin` / `admin`, a dev `SECRET_KEY`, default database credentials, much of the backend privacy functionality removed, and a localhost-only blob store with permissive CORS.

The production deployment used by the hosted service has been **stripped from this release**: the CI/CD pipeline, real secrets and API keys, hosted-service email/legal content, and similar security code are not included. The hardened `ENVIRONMENT=production` settings (DEBUG off, security headers, real Azure/SendGrid wiring) remain in [`settings.py`](vid2scene_server/vid2scene_server/settings.py) for reference only.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party attribution.

`privacy.html` and `terms.html` ship as **placeholder stubs** (with `[DATE]` / `[CONTACT EMAIL]` markers) — replace them with your own privacy policy and terms of use before running a public instance.
