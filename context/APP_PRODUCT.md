# App / web product — vision, UX, and feature roadmap

**Purpose:** In-depth, organized spec for the **California beach water-quality consumer experience** (map, forecasts, copy, design system, competitive positioning). This is separate from **`plan.md`**, which stays focused on **Bayesian modeling, data, and inference**.

**Stack sketch (from brainstorm):** Next.js + MapLibre + deck.gl on Vercel; static predictions updated by the offline pipeline; no heavy backend for v1.

**Related:** `PROJECT_BRIEF.md` (science backbone), `DECISIONS.md`, `INTERFACES.md` (model outputs → API shapes), `DATASETS.md`. **Mockups / fake demo data (pipeline summary + JSON examples):** `MODEL_OUTPUTS_FOR_DEMO.md`.

---

## 1. Product in one sentence

A **California-coast beach water-quality** experience that answers in **~5 seconds** whether the ocean is **likely safe to swim**, with a **seven-day outlook**, for **surfers and beach-goers** — not regulators.

**Mental model:** Apple Weather × Surfline × PurpleAir, but for **water bacteria** (and, in later iterations, surf/weather context layered on).

---

## 2. Core jobs-to-be-done

| Job | Output the user wants |
|-----|------------------------|
| Tell me if the beach I’m about to drive to is safe | One color, one sentence |
| Pick the safest beach near me right now | Ranked list of nearby beaches |
| Is this weekend OK at my regular spot? | 7-day outlook for one beach |
| Why is the water bad today? | Plain-language explanation (rain, runoff, etc.) |
| Notify me when my beach goes from yellow back to green | Push / email |
| Compare this beach to alternatives | Side-by-side |
| I want to understand the data | Model transparency / methodology |
| I want the raw numbers | Stats mode (advanced, opt-in) |

### Explicit non-goals (for now)

- Replace **official county advisories**
- Predict **surf height** (defer to Surfline) or **weather** (defer to Apple Weather / Open-Meteo)
- Provide **medical advice**

---

## 3. Personas (priority for UI)

| Code | Who | Design priority |
|------|-----|-----------------|
| **A** Daily Daniel — dawn-patrol surfer | Primary | High |
| **B** Routine Rachel — open-water swimmer | Primary | High |
| **C** Weekend Family Mike & Anna | High volume | High |
| **E** Tourist Tom | High acquisition | High |
| **D** Triathlete Tina | Probabilistic planning | Power features |
| **F** Lifeguard / municipal | Drill-down, confidence | Power features |
| **G** Researcher / journalist | History, downloads, methodology | Power features |
| **H** Local environmental advocate | Shareable viz, YoY | Power features |

**Rule:** Optimize home map + list for **A, B, C, E**. Put **stats mode, exports, methodology** behind toggles so **D, F, G, H** are served without cluttering the default path.

---

## 4. Day-in-life scenarios (abbreviated)

1. **Friday dawn:** Open app → pinned beach (e.g. El Porto) green → 3-second decision.
2. **After a storm:** Push → beach red → user sees nearby yellow alternative on map.
3. **Family Saturday:** Wed check of 7-day → Sat re-check → drive with kids.
4. **Tourist:** ZIP search → top 5 beaches by status + distance → detail with photo + plain language.
5. **South Bay sewage event:** Map shows regional pattern; detail links IBWC / context copy.
6. **Researcher:** Stats mode → download posteriors for a window → cite methodology page.
7. **Advocate:** History view → YoY red days → share screenshot.

---

## 5. Reference apps — what to learn from

| App | Steal | Gap / your wedge |
|-----|-------|------------------|
| Heal the Bay Beach Report Card | Trust, statewide CA, simple grades | Dated UX, weekly, no probabilistic daily forecast |
| Swim Guide | Map-first, traffic-light | No CA-tuned forecast |
| CA SWRCB portal | Authority | Not a consumer product |
| Surfline | 7-day polish, surfer mental model | Water quality not core |
| PurpleAir | Map UX, color depth | Different domain — same interaction patterns |

**Adjacent UX:** Apple Weather (hero + strip), Windy (map overlays), AllTrails (map + list), Yelp (search → detail).

---

## 6. Competitive positioning (2×2 sketch)

- **Axis:** forecast depth (low ↔ high) vs surf/consumer-friendly ↔ health/science-heavy.
- **Target quadrant:** **High-res forecast** × **surf/consumer-friendly** — probabilistic **daily** CA-specific outlook with app polish; **Heal the Bay / Swim Guide / portals** sit in other quadrants (current-only or regulator tone).

**One-line wedge:** Forecast-grade **uncertainty-aware** water-quality + consumer UX; nothing in CA fully occupies that box today.

---

## 7. Design language — “Coastal Modernist”

**Vibe:** Patagonia × Apple Weather × quiet surf shop. Restrained, ocean-aware. No tropical clichés.

### Palette (v1 lock)

**Light:** Background `#FBF8F1`, surface `#FFFFFF`, text `#0F172A`, primary `#0E7490`, accent `#F97316` (CTAs).

**Dark:** Background `#0A1929`, surface `#0F2438`, text `#E2E8F0`, primary `#22D3EE`, accent `#FB923C`.

**Risk:** Safe `#10B981`, caution `#F59E0B`, avoid `#EF4444`, no data `#94A3B8` — **always pair with icon** (never color-only).

### Typography & motion

- Headings: Fraunces or Geist; body: Inter or Geist Sans; numbers: Geist Mono / JetBrains Mono — **two families max**.
- Icons: Phosphor or Lucide; soft, no skeuomorphism on icons.
- Motion: subtle; respect `prefers-reduced-motion`.
- **Voice:** Plain, calm, editorial — *“We’d skip it today”* not *“Enterococcus exceedance probability 0.78.”*

---

## 8. UX architecture — screens

| Screen | Job |
|--------|-----|
| **Home (map)** | Full-bleed CA coast, colored dots; bottom sheet list by status × distance; search; location FAB; optional heatmap |
| **Beach detail** | Hero photo, big status + one-line why, 7-day strip, collapsible posterior/detail, “Why?” features, recent samples, alternatives, **persistent disclaimer** |
| **Search / list** | ZIP, city, near me; sort distance / status / name |
| **Compare** | 2–4 beaches, overlaid 7-day |
| **Watchlist** | Pinned + alerts — **v2** (sign-in) |
| **About / methodology** | Who we are, how the model works, disclaimer, data credits (SWRCB, NOAA, CDIP, SCCOOS…), contact |
| **Stats mode** | Posteriors, calibration, metadata — **hidden default** |

---

## 9. Health-adjacent responsibility

- Persistent disclaimer: **not an official advisory**; defer to county postings.
- Link **authority** pages (county DPH) from every beach.
- Surface **uncertainty** when data are stale or sparse.
- No medical claims — “elevated risk” language.
- **Methodology** public (or open model).
- **Spanish** — v1 or fast-follow; **WCAG AA** on status colors; icons + color.

---

## 10. Naming

**DataTide** = project / science name. **Consumer app** gets a friendlier name later (brainstorm: Tideline, Lineup, Brine, Coastline, First Light, Greenwater…). **Ship with a placeholder** until design language is locked.

---

## 11. Future / risks (summary)

| Risk | Mitigation |
|------|------------|
| User ill after “green” | Disclaimer + uncertainty + official links |
| Model / pipeline stale | Banner: last update time |
| Traffic / map costs | MapLibre + free tiles; Cloudflare if needed |
| Mobile-first | **PWA** first; native later |
| Spanish | Localize early |
| Color-blind | Icon + color |

**TL;DR product promise:** CA beach water-quality **forecast** (7 days), **plain language**, **honest uncertainty**, **coastal modernist** UI, **static-first** delivery, **ethics**: not an advisory.

---

## 12. Feature additions beyond bacteria — parity & differentiation

**Idea:** Bacteria forecasts are the **science moat**; **Open-Meteo** (free, keyless, client-callable) closes the gap to Surfline-class **surf/weather** context without owning a weather model.

### Benchmark: what surf apps ship

Wave height/period/direction, wind, tide high/low, SST, air temp/UV, star rating, sunrise/sunset, webcams (often paywalled), maps.

### Open-Meteo endpoints (representative)

| Endpoint | Use |
|----------|-----|
| `api.open-meteo.com/v1/forecast` | Weather, wind, UV, precip |
| `marine-api.open-meteo.com/v1/marine` | Waves, swell, SST |
| `air-quality-api.open-meteo.com/v1/air-quality` | AQI, PM |
| `archive-api.open-meteo.com/v1/archive` | Historical weather |

**Client-only:** pass beach `lat`/`lon` on load — **no pipeline change** for v1 of these layers.

### Tier 1 — quick wins (mostly client)

Sunrise/sunset/golden hour (**suncalc**), moon phase, tide **chart** from existing tide data, today’s weather (one call), wind + **onshore/offshore** heuristic, SST (marine + SCCOOS where you already join), UV, AQI, **favorites** (`localStorage`), **share** (`navigator.share`), maps deep link, **dark mode**, **PWA** manifest, **last updated** from pipeline meta, QR/copy link.

### Tier 2 — parity & differentiator

- **Open-Meteo Marine** — 7-day wave/swell aligned with bacteria strip.
- **Combined conditions score (1–5 stars):** weighted blend of **water quality** (e.g. 1 − p_exceed), **surf quality** (Hs/Tp heuristic), **wind quality** (offshore vs onshore). Weights by persona (surfer vs swimmer vs family) — **novel vs Surfline + Swim Guide**.
- **OSM amenities** (Overpass at build time): parking, toilets, showers, lifeguard, dog-friendly → `beaches.json`.
- **“Best time to go today”** — 24h strip from tide × UV × wind × simple crowd heuristic.
- **“Conditions like this”** — historical pattern text from your Parquet (trust).
- **Wind/swell particles** on map (deck.gl) — optional polish.

### Tier 3 — polish

Weekly digest email (e.g. Resend), red tide / HAB overlay (CDPH biotoxin public data), rip-current NWS zones, webcam **links**, Spanish (`next-intl`), PWA push, Unsplash hero images, a11y audit, Wallet pass (fun).

### Do **not** build (traps)

Own weather model; crowd prediction without data; **accounts** for v1 if `localStorage` suffices; forums/comments; live chat; unreliable hardware; **native apps day 1**; LLM narratives at request time for v1 (prefer templates).

### Suggested “polish sprint” narrative (post–v1)

**Week 1:** Open-Meteo weather + wind + wave + SST + UV on every beach page; sun/moon; tide chart; favorites; PWA; dark mode; share.

**Week 2:** Combined score + OSM amenities + “best time” strip + historical “like this” + digest.

**Positioning shift**

- **Before:** “California beach bacteria forecast.”
- **After:** “The only place that combines **clean water**, **surf**, and **wind** in one **honest** 7-day view — built for California.”

---

## 13. How this doc relates to `plan.md`

| Document | Focus |
|----------|--------|
| **`plan.md`** | Bayesian model, likelihoods, HSGP, SVI/NUTS, data critique |
| **`APP_PRODUCT.md`** (this file) | Who uses the app, what screens ship, design/ethics, Open-Meteo parity roadmap |

Model outputs (posteriors, exceedance probs) **feed** the beach detail UI and copy templates; **INTERFACES.md** should stay aligned with whatever the API exports.

---

*Last organized from team brainstorm + “easy additions” memo. Update this file when scope or v1 cut line changes.*
