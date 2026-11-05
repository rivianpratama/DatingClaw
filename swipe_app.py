from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import socket
import sys
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageDraw
import uvicorn

from crop_54k_dataset import DEFAULT_CROP_BOX, clamp_int, is_what_a_match, get_match_template

app = FastAPI(title="BumbleClaw Dating-App Swipe Labeler")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SOURCE_DIR = Path(r"D:\BumbleLog")
CROPPED_DIR = Path(r"D:\BumbleLog\cropped")
LABELS_CSV = Path(r"D:\BumbleLog\binary_swipe_labels.csv")
SCORES_CSV = Path(r"D:\BumbleLog\scores.csv")
SERVER_PORT = 7865

CSV_FIELDS = ["timestamp", "filename", "path", "action", "rating_1_5", "rating"]

# In-memory queue state protected by lock
STATE_LOCK = threading.Lock()
ALL_TASKS: list[str] = []
LABELED_SET: set[str] = set()
UNDO_STACK: list[dict[str, Any]] = []
STATS = {"likes": 0, "nopes": 0}


def get_lan_ips(port: int) -> list[str]:
    urls = []
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                urls.append(f"http://{ip}:{port}")
    except Exception:
        pass
    if not urls:
        urls.append(f"http://localhost:{port}")
    return urls


def load_dataset() -> None:
    global ALL_TASKS, LABELED_SET, STATS, UNDO_STACK
    with STATE_LOCK:
        ALL_TASKS = []
        LABELED_SET = set()
        UNDO_STACK = []
        STATS = {"likes": 0, "nopes": 0}

        # Load blacklist of purged match screens if present
        blacklist = set()
        match_txt = Path("match_screens.txt")
        if match_txt.exists():
            with open(match_txt, encoding="utf-8") as f:
                blacklist = set(line.strip() for line in f if line.strip())

        # 1. Discover tasks in order from scores.csv
        seen = set()
        if SCORES_CSV.exists():
            with open(SCORES_CSV, mode="r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    s = (row.get("screenshot") or "").strip()
                    if s and s not in seen and s not in blacklist:
                        if (CROPPED_DIR / s).exists() or (SOURCE_DIR / s).exists():
                            seen.add(s)
                            ALL_TASKS.append(s)

        # Also find any remaining image files in SOURCE_DIR
        valid_exts = {".webp", ".jpg", ".jpeg", ".png"}
        if SOURCE_DIR.exists():
            for p in sorted(SOURCE_DIR.iterdir()):
                if p.is_file() and p.suffix.lower() in valid_exts and p.name not in seen and p.name not in blacklist:
                    seen.add(p.name)
                    ALL_TASKS.append(p.name)

        # 2. Read existing labels if any
        if LABELS_CSV.exists():
            with open(LABELS_CSV, mode="r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    fname = (row.get("filename") or "").strip()
                    action = (row.get("action") or "").strip()
                    if fname:
                        LABELED_SET.add(fname)
                        UNDO_STACK.append(row)
                        if action == "right":
                            STATS["likes"] += 1
                        elif action == "left":
                            STATS["nopes"] += 1

        print(f"Loaded {len(ALL_TASKS)} total profiles. Already labeled: {len(LABELED_SET)} (Likes: {STATS['likes']}, Nopes: {STATS['nopes']})")


def get_unlabeled_queue() -> list[str]:
    return [task for task in ALL_TASKS if task not in LABELED_SET]


@app.on_event("startup")
def on_startup() -> None:
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    load_dataset()


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    with STATE_LOCK:
        queue = get_unlabeled_queue()
        total = len(ALL_TASKS)
        labeled = len(LABELED_SET)
        return {
            "total": total,
            "labeled": labeled,
            "remaining": len(queue),
            "likes": STATS["likes"],
            "nopes": STATS["nopes"],
            "like_rate": round((STATS["likes"] / labeled) * 100) if labeled > 0 else 0,
            "can_undo": len(UNDO_STACK) > 0,
        }


@app.get("/api/cards")
def api_cards(count: int = 5) -> dict[str, Any]:
    with STATE_LOCK:
        queue = get_unlabeled_queue()
        selected = queue[:count]
        cards = [
            {
                "id": filename,
                "filename": filename,
                "image_url": f"/images/cropped/{filename}",
            }
            for filename in selected
        ]
        return {
            "cards": cards,
            "total": len(ALL_TASKS),
            "labeled": len(LABELED_SET),
            "remaining": len(queue),
        }


@app.post("/api/swipe")
def api_swipe(payload: dict[str, Any]) -> dict[str, Any]:
    filename = str(payload.get("filename") or "").strip()
    action = str(payload.get("action") or "").strip().lower()

    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")
    if action not in ("left", "right"):
        raise HTTPException(status_code=400, detail="action must be 'left' or 'right'")

    with STATE_LOCK:
        if filename in LABELED_SET:
            # Already swiped
            pass
        else:
            LABELED_SET.add(filename)

        rating_1_5 = 5 if action == "right" else 1
        rating = 100 if action == "right" else 0
        now_iso = datetime.now(timezone.utc).isoformat()
        full_path = (SOURCE_DIR / filename).resolve().as_posix()

        row = {
            "timestamp": now_iso,
            "filename": filename,
            "path": full_path,
            "action": action,
            "rating_1_5": rating_1_5,
            "rating": rating,
        }

        # Atomic append to CSV
        LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
        write_header = not LABELS_CSV.exists() or LABELS_CSV.stat().st_size == 0
        with open(LABELS_CSV, mode="a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        UNDO_STACK.append(row)
        if action == "right":
            STATS["likes"] += 1
        else:
            STATS["nopes"] += 1

        queue = get_unlabeled_queue()
        labeled_count = len(LABELED_SET)
        return {
            "success": True,
            "labeled": labeled_count,
            "remaining": len(queue),
            "likes": STATS["likes"],
            "nopes": STATS["nopes"],
            "like_rate": round((STATS["likes"] / labeled_count) * 100) if labeled_count > 0 else 0,
            "can_undo": len(UNDO_STACK) > 0,
        }


@app.post("/api/undo")
def api_undo() -> dict[str, Any]:
    with STATE_LOCK:
        if not UNDO_STACK:
            raise HTTPException(status_code=400, detail="Nothing to undo")

        last_row = UNDO_STACK.pop()
        filename = last_row["filename"]
        action = last_row.get("action", "")

        if filename in LABELED_SET:
            LABELED_SET.remove(filename)

        if action == "right":
            STATS["likes"] = max(0, STATS["likes"] - 1)
        elif action == "left":
            STATS["nopes"] = max(0, STATS["nopes"] - 1)

        # Rewrite CSV without the last row
        if LABELS_CSV.exists():
            rows = []
            with open(LABELS_CSV, mode="r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            # Remove the last matching row
            if rows and rows[-1].get("filename") == filename:
                rows.pop()
            else:
                # Filter out the specific occurrence
                for i in range(len(rows) - 1, -1, -1):
                    if rows[i].get("filename") == filename:
                        rows.pop(i)
                        break

            with open(LABELS_CSV, mode="w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

        restored_card = {
            "id": filename,
            "filename": filename,
            "image_url": f"/images/cropped/{filename}",
        }

        queue = get_unlabeled_queue()
        labeled_count = len(LABELED_SET)
        return {
            "success": True,
            "restored_card": restored_card,
            "labeled": labeled_count,
            "remaining": len(queue),
            "likes": STATS["likes"],
            "nopes": STATS["nopes"],
            "like_rate": round((STATS["likes"] / labeled_count) * 100) if labeled_count > 0 else 0,
            "can_undo": len(UNDO_STACK) > 0,
        }


@app.get("/images/cropped/{filename}")
def get_cropped_image(filename: str) -> Response:
    cropped_path = CROPPED_DIR / filename
    if cropped_path.exists() and cropped_path.stat().st_size > 0:
        media_type = "image/webp" if cropped_path.suffix.lower() == ".webp" else "image/jpeg"
        return FileResponse(cropped_path, media_type=media_type)

    raw_path = SOURCE_DIR / filename
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # On-the-fly fallback crop if background pre-crop hasn't reached this file yet
    try:
        with Image.open(raw_path) as image:
            image = image.convert("RGB")
            w, h = image.size
            l = clamp_int(w * DEFAULT_CROP_BOX[0], 0, w - 1)
            t = clamp_int(h * DEFAULT_CROP_BOX[1], 0, h - 1)
            r = clamp_int(w * DEFAULT_CROP_BOX[2], l + 1, w)
            b = clamp_int(h * DEFAULT_CROP_BOX[3], t + 1, h)
            cropped = image.crop((l, t, r, b))

            cropped_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = "WEBP" if cropped_path.suffix.lower() == ".webp" else "JPEG"
            cropped.save(cropped_path, format=fmt, quality=90, optimize=True)

            media_type = "image/webp" if fmt == "WEBP" else "image/jpeg"
            return FileResponse(cropped_path, media_type=media_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Crop failure: {exc}")


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>BumbleClaw Swipe</title>
  <style>
    :root {
      --bg: #0d0f14;
      --card-bg: #161822;
      --text: #ffffff;
      --text-muted: #8e94a5;
      --like: #00f59b;
      --nope: #ff3b56;
      --undo: #ffb800;
      --card-radius: 28px;
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-tap-highlight-color: transparent;
      user-select: none;
      -webkit-user-select: none;
    }
    html, body {
      width: 100%;
      height: 100%;
      overflow: hidden;
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
    }
    #app {
      display: flex;
      flex-direction: column;
      width: 100vw;
      height: 100dvh;
      padding: max(6px, env(safe-area-inset-top)) 8px max(8px, env(safe-area-inset-bottom));
      justify-content: space-between;
      align-items: center;
      position: relative;
    }

    /* Top Bar / Progress Pill */
    header {
      width: 100%;
      max-width: 440px;
      display: flex;
      justify-content: center;
      align-items: center;
      z-index: 50;
      height: 32px;
      margin-bottom: 4px;
    }
    .status-pill {
      background: rgba(255, 255, 255, 0.08);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 999px;
      padding: 5px 14px;
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.3px;
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .status-pill .accent {
      color: var(--like);
    }
    .status-pill .dot {
      color: rgba(255, 255, 255, 0.3);
    }

    /* Card Deck Container */
    #deck-container {
      flex: 1;
      width: 100%;
      max-width: 440px;
      position: relative;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 0;
      margin: 0;
      perspective: 1000px;
    }

    .card {
      position: absolute;
      height: 100%;
      max-height: 100%;
      aspect-ratio: 516 / 935;
      max-width: 100%;
      background: var(--card-bg);
      border-radius: var(--card-radius);
      overflow: hidden;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6);
      touch-action: none;
      transform-origin: center bottom;
      will-change: transform, opacity;
      cursor: grab;
      border: 1px solid rgba(255, 255, 255, 0.08);
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .card:active {
      cursor: grabbing;
    }

    .card img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      pointer-events: none;
      background: #0d0f14;
      border-radius: var(--card-radius);
    }

    .card-gradient {
      display: none;
    }

    /* Swipe Badges / Stamps */
    .stamp {
      position: absolute;
      top: 36px;
      padding: 6px 18px;
      font-size: 32px;
      font-weight: 900;
      letter-spacing: 2px;
      border-radius: 12px;
      border: 4px solid;
      pointer-events: none;
      opacity: 0;
      transform: scale(0.9);
      transition: opacity 0.05s ease;
      z-index: 20;
    }
    .stamp.like {
      left: 28px;
      color: var(--like);
      border-color: var(--like);
      transform: rotate(-18deg);
      text-shadow: 0 0 16px rgba(0, 245, 155, 0.4);
    }
    .stamp.nope {
      right: 28px;
      color: var(--nope);
      border-color: var(--nope);
      transform: rotate(18deg);
      text-shadow: 0 0 16px rgba(255, 59, 86, 0.4);
    }

    /* Floating Bottom Action Bar */
    footer {
      width: 100%;
      max-width: 440px;
      display: flex;
      justify-content: space-around;
      align-items: center;
      padding: 4px 20px;
      z-index: 50;
      height: 64px;
    }
    .btn {
      border: none;
      outline: none;
      border-radius: 50%;
      display: flex;
      justify-content: center;
      align-items: center;
      cursor: pointer;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
      transition: transform 0.12s cubic-bezier(0.2, 1, 0.4, 1), background-color 0.12s;
    }
    .btn:active {
      transform: scale(0.88);
    }
    .btn-undo {
      width: 52px;
      height: 52px;
      background: rgba(255, 184, 0, 0.14);
      color: var(--undo);
      border: 1px solid rgba(255, 184, 0, 0.25);
    }
    .btn-undo:disabled {
      opacity: 0.3;
      cursor: not-allowed;
      transform: none;
    }
    .btn-nope {
      width: 68px;
      height: 68px;
      background: rgba(255, 59, 86, 0.14);
      color: var(--nope);
      border: 1.5px solid rgba(255, 59, 86, 0.28);
    }
    .btn-like {
      width: 68px;
      height: 68px;
      background: rgba(0, 245, 155, 0.14);
      color: var(--like);
      border: 1.5px solid rgba(0, 245, 155, 0.28);
    }

    /* Empty / Completion screen */
    #empty-state {
      display: none;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      gap: 16px;
      text-align: center;
      padding: 40px;
    }
    #empty-state h2 {
      font-size: 28px;
      font-weight: 800;
    }
    #empty-state p {
      color: var(--text-muted);
      font-size: 15px;
    }
  </style>
</head>
<body>
  <div id="app">
    <header>
      <div class="status-pill">
        <span id="counter-text">Loading...</span>
        <span class="dot">·</span>
        <span id="like-rate" class="accent">0% Like</span>
      </div>
    </header>

    <main id="deck-container">
      <div id="empty-state">
        <h2>🎉 All Swiped!</h2>
        <p>You have finished swiping through all available profiles.</p>
      </div>
    </main>

    <footer>
      <button id="undo-btn" class="btn btn-undo" title="Undo Last Swipe (Z)" disabled>
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 7v6h6"></path>
          <path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"></path>
        </svg>
      </button>

      <button id="nope-btn" class="btn btn-nope" title="Swipe Left / Nope (←)">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>

      <button id="like-btn" class="btn btn-like" title="Swipe Right / Like (→)">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
        </svg>
      </button>
    </footer>
  </div>

  <script>
    const deck = document.getElementById('deck-container');
    const counterText = document.getElementById('counter-text');
    const likeRateText = document.getElementById('like-rate');
    const undoBtn = document.getElementById('undo-btn');
    const nopeBtn = document.getElementById('nope-btn');
    const likeBtn = document.getElementById('like-btn');
    const emptyState = document.getElementById('empty-state');

    let cardQueue = [];
    let isFetching = false;
    let isAnimating = false;
    const preloadedImages = new Set();

    async function fetchMoreCards(count = 5) {
      if (isFetching) return;
      isFetching = true;
      try {
        const res = await fetch(`/api/cards?count=${count}`);
        const data = await res.json();
        
        // Add new cards to queue avoiding duplicates
        const currentIds = new Set(cardQueue.map(c => c.filename));
        for (const card of data.cards) {
          if (!currentIds.has(card.filename)) {
            cardQueue.push(card);
            preloadImage(card.image_url);
          }
        }
        renderDeck();
      } catch (err) {
        console.error("Failed to fetch cards:", err);
      } finally {
        isFetching = false;
      }
    }

    function preloadImage(url) {
      if (preloadedImages.has(url)) return;
      preloadedImages.add(url);
      const img = new Image();
      img.src = url;
    }

    async function updateStats() {
      try {
        const res = await fetch('/api/state');
        const state = await res.json();
        counterText.textContent = `${state.labeled.toLocaleString()} / ${state.total.toLocaleString()}`;
        likeRateText.textContent = `${state.like_rate}% Like`;
        undoBtn.disabled = !state.can_undo;

        if (state.remaining === 0 && cardQueue.length === 0) {
          emptyState.style.display = 'flex';
        } else {
          emptyState.style.display = 'none';
        }
      } catch (e) {
        console.error("Stats update failed:", e);
      }
    }

    function renderDeck() {
      // Remove stale cards
      const existingCards = deck.querySelectorAll('.card');
      existingCards.forEach(c => c.remove());

      if (cardQueue.length === 0) {
        if (!isFetching) fetchMoreCards(5);
        return;
      }

      // Render top 3 cards in reverse order (so 0 is on top)
      const visible = cardQueue.slice(0, 3);
      for (let i = visible.length - 1; i >= 0; i--) {
        const cardData = visible[i];
        const cardEl = createCardElement(cardData, i);
        deck.appendChild(cardEl);
      }
    }

    function createCardElement(data, index) {
      const card = document.createElement('div');
      card.className = 'card';
      card.dataset.filename = data.filename;

      // Card scale & offset for stacked depth
      const scale = 1 - (index * 0.04);
      const translateY = index * 12;
      card.style.transform = `translate3d(0, ${translateY}px, 0) scale(${scale})`;
      card.style.zIndex = 10 - index;
      card.style.opacity = index > 2 ? 0 : 1;

      card.innerHTML = `
        <img src="${data.image_url}" alt="Profile" loading="eager" draggable="false">
        <div class="card-gradient"></div>
        <div class="stamp like">LIKE</div>
        <div class="stamp nope">NOPE</div>
      `;

      if (index === 0) {
        initDrag(card, data);
      }

      return card;
    }

    function initDrag(card, data) {
      let isDragging = false;
      let startX = 0, startY = 0;
      let currentX = 0, currentY = 0;

      const likeStamp = card.querySelector('.stamp.like');
      const nopeStamp = card.querySelector('.stamp.nope');

      function onPointerDown(e) {
        if (isAnimating) return;
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        currentX = 0;
        currentY = 0;
        card.setPointerCapture(e.pointerId);
        card.style.transition = 'none';
      }

      function onPointerMove(e) {
        if (!isDragging) return;
        currentX = e.clientX - startX;
        currentY = e.clientY - startY;

        const rotation = (currentX / 300) * 16;
        card.style.transform = `translate3d(${currentX}px, ${currentY}px, 0) rotate(${rotation}deg)`;

        // Dynamic stamp opacity
        const swipeRatio = currentX / 100;
        if (swipeRatio > 0) {
          likeStamp.style.opacity = Math.min(Math.max((swipeRatio - 0.15) * 1.5, 0), 1);
          nopeStamp.style.opacity = 0;
        } else {
          nopeStamp.style.opacity = Math.min(Math.max((-swipeRatio - 0.15) * 1.5, 0), 1);
          likeStamp.style.opacity = 0;
        }
      }

      function onPointerUp(e) {
        if (!isDragging) return;
        isDragging = false;
        try { card.releasePointerCapture(e.pointerId); } catch (_) {}

        const threshold = 95;
        if (currentX > threshold) {
          executeSwipe('right', card, data);
        } else if (currentX < -threshold) {
          executeSwipe('left', card, data);
        } else {
          // Snap back
          card.style.transition = 'transform 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
          card.style.transform = 'translate3d(0, 0, 0) rotate(0deg)';
          likeStamp.style.opacity = 0;
          nopeStamp.style.opacity = 0;
        }
      }

      card.addEventListener('pointerdown', onPointerDown);
      card.addEventListener('pointermove', onPointerMove);
      card.addEventListener('pointerup', onPointerUp);
      card.addEventListener('pointercancel', onPointerUp);
    }

    async function executeSwipe(direction, cardEl, data) {
      if (isAnimating) return;
      isAnimating = true;

      // Haptic feedback if available
      if (navigator.vibrate) navigator.vibrate(12);

      // Trigger fly-off animation
      const flyX = direction === 'right' ? window.innerWidth * 1.3 : -window.innerWidth * 1.3;
      const flyRotation = direction === 'right' ? 28 : -28;

      cardEl.style.transition = 'transform 0.32s cubic-bezier(0.2, 0.9, 0.3, 1), opacity 0.32s ease';
      cardEl.style.transform = `translate3d(${flyX}px, 0, 0) rotate(${flyRotation}deg)`;
      cardEl.style.opacity = 0;

      // Animate card underneath to top position
      const nextCard = deck.querySelectorAll('.card')[1];
      if (nextCard) {
        nextCard.style.transition = 'transform 0.28s cubic-bezier(0.2, 0.9, 0.3, 1)';
        nextCard.style.transform = 'translate3d(0, 0, 0) scale(1)';
      }

      // Optimistically remove from queue
      cardQueue.shift();

      // Submit swipe to server in background
      fetch('/api/swipe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: data.filename, action: direction })
      }).then(res => res.json()).then(res => {
        updateStats();
      }).catch(err => {
        console.error("Swipe failed:", err);
      });

      setTimeout(() => {
        cardEl.remove();
        isAnimating = false;
        renderDeck();
        if (cardQueue.length < 4) {
          fetchMoreCards(5);
        }
      }, 300);
    }

    async function undoSwipe() {
      if (isAnimating || undoBtn.disabled) return;
      isAnimating = true;

      try {
        const res = await fetch('/api/undo', { method: 'POST' });
        const data = await res.json();
        if (data.success && data.restored_card) {
          // Prepend restored card to queue
          cardQueue.unshift(data.restored_card);
          renderDeck();

          // Animate card flying back into deck
          const topCard = deck.querySelector('.card');
          if (topCard) {
            topCard.style.transition = 'none';
            topCard.style.transform = 'translate3d(0, -60px, 0) scale(0.92)';
            topCard.style.opacity = 0;
            requestAnimationFrame(() => {
              topCard.style.transition = 'transform 0.32s cubic-bezier(0.175, 0.885, 0.32, 1.275), opacity 0.32s';
              topCard.style.transform = 'translate3d(0, 0, 0) scale(1)';
              topCard.style.opacity = 1;
            });
          }
          updateStats();
        }
      } catch (err) {
        console.error("Undo failed:", err);
      } finally {
        setTimeout(() => { isAnimating = false; }, 320);
      }
    }

    // Button Listeners
    nopeBtn.addEventListener('click', () => {
      const topCard = deck.querySelector('.card');
      if (topCard && cardQueue[0]) {
        topCard.querySelector('.stamp.nope').style.opacity = 1;
        executeSwipe('left', topCard, cardQueue[0]);
      }
    });

    likeBtn.addEventListener('click', () => {
      const topCard = deck.querySelector('.card');
      if (topCard && cardQueue[0]) {
        topCard.querySelector('.stamp.like').style.opacity = 1;
        executeSwipe('right', topCard, cardQueue[0]);
      }
    });

    undoBtn.addEventListener('click', undoSwipe);

    // Keyboard Shortcuts (Desktop support)
    document.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      if (e.key === 'ArrowLeft') {
        nopeBtn.click();
      } else if (e.key === 'ArrowRight') {
        likeBtn.click();
      } else if (e.key === 'z' || e.key === 'Z' || e.key === 'Backspace') {
        undoBtn.click();
      }
    });

    // Initial Load
    fetchMoreCards(5);
    updateStats();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(content=HTML_CONTENT, status_code=200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dating-App Mobile Swipe Labeler Server")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Port to bind server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    return parser.parse_args()


def print_banner(port: int) -> None:
    urls = get_lan_ips(port)
    print("\n" + "=" * 60)
    print("[*] BumbleClaw Dating-App Mobile Swipe Labeler")
    print("=" * 60)
    print("\n>>> OPEN ON YOUR PHONE (Connected to same Wi-Fi):")
    for url in urls:
        print(f"    --> {url}")
    print("\n>>> DESKTOP CONTROLS:")
    print("    <- ArrowLeft  : Nope / Pass (Score = 0)")
    print("    -> ArrowRight : Like (Score = 100)")
    print("    Z / Backspace: Undo last swipe")
    print("\n" + "=" * 60 + "\n")


def main() -> int:
    args = parse_args()
    print_banner(args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
