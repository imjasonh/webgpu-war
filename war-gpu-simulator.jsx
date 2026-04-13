import { useState, useEffect, useRef, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════════
// WGSL COMPUTE SHADER
// ═══════════════════════════════════════════════════════════════════
const SHADER = `
fn pcg(state: ptr<function, u32>) -> u32 {
  let old = *state;
  *state = old * 747796405u + 2891336453u;
  let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
  return (word >> 22u) ^ word;
}

@group(0) @binding(0) var<storage, read> seeds: array<u32>;
@group(0) @binding(1) var<storage, read_write> results: array<u32>;

const MAX_ROUNDS: u32 = 20000u;
const FIELDS: u32 = 7u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  var rng = seeds[idx];
  let original_seed = seeds[idx];

  // Single array: cards[0..51] = player A's circular buffer
  //               cards[52..103] = player B's circular buffer
  // Initialize as identity deck, then shuffle, then split
  var cards: array<u32, 104>;
  for (var i = 0u; i < 52u; i = i + 1u) { cards[i] = i; }

  // Fisher-Yates shuffle in-place on first 52 slots
  for (var i = 51u; i > 0u; i = i - 1u) {
    let j = pcg(&rng) % (i + 1u);
    let tmp = cards[i]; cards[i] = cards[j]; cards[j] = tmp;
  }

  // Move second half to B's buffer region (slots 52-77)
  for (var i = 0u; i < 26u; i = i + 1u) {
    cards[52u + i] = cards[26u + i];
  }

  // A uses slots 0..51, B uses slots 52..103
  var ah: u32 = 0u; var at_: u32 = 26u; var ac: u32 = 26u;
  var bh: u32 = 0u; var bt_: u32 = 26u; var bc: u32 = 26u;

  var tr: u32 = 0u;
  var wc: u32 = 0u;
  var win: u32 = 2u;
  var sa: u32 = 0u; var sb: u32 = 0u;
  var msa: u32 = 0u; var msb: u32 = 0u;
  var bw: u32 = 0u;
  var done: bool = false;

  var pot: array<u32, 52>;

  for (var r = 0u; r < MAX_ROUNDS; r = r + 1u) {
    if (done) { break; }
    if (ac == 0u) { win = 1u; tr = r; break; }
    if (bc == 0u) { win = 0u; tr = r; break; }

    let ca = cards[ah % 52u]; ah = (ah + 1u) % 52u; ac = ac - 1u;
    let cb = cards[52u + (bh % 52u)]; bh = (bh + 1u) % 52u; bc = bc - 1u;

    var pc_: u32 = 2u;
    pot[0] = ca; pot[1] = cb;

    var wa: u32 = ca / 4u;
    var wb: u32 = cb / 4u;
    var wd: u32 = 0u;

    for (var w = 0u; w < 15u; w = w + 1u) {
      if (wa != wb || done) { break; }
      wc = wc + 1u; wd = wd + 1u;

      let af = min(3u, ac);
      for (var f = 0u; f < af; f = f + 1u) {
        pot[pc_] = cards[ah % 52u]; pc_ = pc_ + 1u; ah = (ah + 1u) % 52u; ac = ac - 1u;
      }
      let bf = min(3u, bc);
      for (var f = 0u; f < bf; f = f + 1u) {
        pot[pc_] = cards[52u + (bh % 52u)]; pc_ = pc_ + 1u; bh = (bh + 1u) % 52u; bc = bc - 1u;
      }

      if (ac == 0u && bc == 0u) { win = 2u; tr = r; done = true; break; }
      if (ac == 0u) { win = 1u; tr = r; done = true; break; }
      if (bc == 0u) { win = 0u; tr = r; done = true; break; }

      let na = cards[ah % 52u]; ah = (ah + 1u) % 52u; ac = ac - 1u;
      let nb = cards[52u + (bh % 52u)]; bh = (bh + 1u) % 52u; bc = bc - 1u;
      pot[pc_] = na; pc_ = pc_ + 1u;
      pot[pc_] = nb; pc_ = pc_ + 1u;
      wa = na / 4u; wb = nb / 4u;
    }

    if (wd > bw) { bw = wd; }

    if (!done) {
      for (var si = pc_ - 1u; si > 0u; si = si - 1u) {
        let sj = pcg(&rng) % (si + 1u);
        let stmp = pot[si]; pot[si] = pot[sj]; pot[sj] = stmp;
      }
      if (wa > wb) {
        for (var p = 0u; p < pc_; p = p + 1u) { cards[at_ % 52u] = pot[p]; at_ = (at_ + 1u) % 52u; ac = ac + 1u; }
        sa = sa + 1u; if (sa > msa) { msa = sa; } sb = 0u;
      } else if (wb > wa) {
        for (var p = 0u; p < pc_; p = p + 1u) { cards[52u + (bt_ % 52u)] = pot[p]; bt_ = (bt_ + 1u) % 52u; bc = bc + 1u; }
        sb = sb + 1u; if (sb > msb) { msb = sb; } sa = 0u;
      }
    }
    if (r == MAX_ROUNDS - 1u) { win = 2u; tr = MAX_ROUNDS; }
  }

  let base = idx * FIELDS;
  results[base] = win;
  results[base + 1u] = tr;
  results[base + 2u] = wc;
  results[base + 3u] = max(msa, msb);
  results[base + 4u] = bw;
  results[base + 5u] = original_seed;
  results[base + 6u] = ac;
}
`;

// ═══════════════════════════════════════════════════════════════════
// CPU GAME LOGIC (shared by benchmark and replay)
// ═══════════════════════════════════════════════════════════════════
function pcg(state) {
  let s = state[0] >>> 0;
  state[0] = (Math.imul(s, 747796405) + 2891336453) >>> 0;
  let word = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}
const RANKS = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"];
const SUITS = ["\u2663","\u2666","\u2665","\u2660"];
function cardName(c) { return RANKS[Math.floor(c / 4)] + SUITS[c % 4]; }
function cardRank(c) { return Math.floor(c / 4); }

// Fast CPU sim — returns summary only (for benchmark)
function cpuSimGame(seed) {
  const state = [seed >>> 0];
  const deck = Array.from({ length: 52 }, (_, i) => i);
  for (let i = 51; i > 0; i--) { const j = pcg(state) % (i + 1); [deck[i], deck[j]] = [deck[j], deck[i]]; }

  const hand = new Uint8Array(104); // circular buffers for both players
  let ah = 0, at = 26, ac = 26, bh = 0, bt = 26, bc = 26;
  for (let i = 0; i < 26; i++) { hand[i] = deck[i]; hand[52 + i] = deck[i + 26]; }

  let tr = 0, wc = 0, winner = 2, msa = 0, msb = 0, sa = 0, sb = 0, bw = 0;
  const pot = new Uint8Array(52);

  for (let r = 0; r < 20000; r++) {
    if (ac === 0) { winner = 1; tr = r; break; }
    if (bc === 0) { winner = 0; tr = r; break; }

    const ca = hand[ah % 52]; ah = (ah + 1) % 52; ac--;
    const cb = hand[52 + (bh % 52)]; bh = (bh + 1) % 52; bc--;
    pot[0] = ca; pot[1] = cb;
    let pc = 2, wa = ca >> 2, wb = cb >> 2, wd = 0, done = false;

    while (wa === wb && !done) {
      wc++; wd++;
      const af = Math.min(3, ac);
      for (let f = 0; f < af; f++) { pot[pc++] = hand[ah % 52]; ah = (ah + 1) % 52; ac--; }
      const bf = Math.min(3, bc);
      for (let f = 0; f < bf; f++) { pot[pc++] = hand[52 + (bh % 52)]; bh = (bh + 1) % 52; bc--; }
      if (ac === 0 && bc === 0) { winner = 2; tr = r; done = true; break; }
      if (ac === 0) { winner = 1; tr = r; done = true; break; }
      if (bc === 0) { winner = 0; tr = r; done = true; break; }
      const na = hand[ah % 52]; ah = (ah + 1) % 52; ac--;
      const nb = hand[52 + (bh % 52)]; bh = (bh + 1) % 52; bc--;
      pot[pc++] = na; pot[pc++] = nb;
      wa = na >> 2; wb = nb >> 2;
    }
    if (wd > bw) bw = wd;
    if (!done) {
      for (let si = pc - 1; si > 0; si--) { const sj = pcg(state) % (si + 1); const tmp = pot[si]; pot[si] = pot[sj]; pot[sj] = tmp; }
      if (wa > wb) {
        for (let p = 0; p < pc; p++) { hand[at % 52] = pot[p]; at = (at + 1) % 52; ac++; }
        sa++; if (sa > msa) msa = sa; sb = 0;
      } else {
        for (let p = 0; p < pc; p++) { hand[52 + (bt % 52)] = pot[p]; bt = (bt + 1) % 52; bc++; }
        sb++; if (sb > msb) msb = sb; sa = 0;
      }
    }
    if (r === 19999) { winner = 2; tr = 20000; }
  }
  return { winner, tr, wc, ms: Math.max(msa, msb), bw, seed };
}

// Full replay with round-by-round state (circular buffers matching GPU)
function replayGame(seed) {
  const state = [seed >>> 0];
  const deck = Array.from({ length: 52 }, (_, i) => i);
  for (let i = 51; i > 0; i--) { const j = pcg(state) % (i + 1); [deck[i], deck[j]] = [deck[j], deck[i]]; }

  // Mirror the GPU: cards[0..51] = A, cards[52..103] = B
  const cards = new Array(104);
  for (let i = 0; i < 26; i++) { cards[i] = deck[i]; cards[52 + i] = deck[26 + i]; }

  let ah = 0, at_ = 26, ac = 26;
  let bh = 0, bt_ = 26, bc = 26;
  const rounds = [];
  const pot = new Array(52);

  for (let r = 0; r < 20000; r++) {
    if (ac === 0) { rounds.push({ round: r, type: "end", winner: "B", aCount: 0, bCount: bc }); break; }
    if (bc === 0) { rounds.push({ round: r, type: "end", winner: "A", aCount: ac, bCount: 0 }); break; }

    const ca = cards[ah % 52]; ah = (ah + 1) % 52; ac--;
    const cb = cards[52 + (bh % 52)]; bh = (bh + 1) % 52; bc--;
    pot[0] = ca; pot[1] = cb;
    let pc = 2;
    let wa = cardRank(ca), wb = cardRank(cb), warDepth = 0, warCards = [], gameOver = false;

    while (wa === wb && !gameOver) {
      warDepth++;
      const fd = [];
      const af = Math.min(3, ac);
      for (let f = 0; f < af; f++) {
        const c = cards[ah % 52]; pot[pc++] = c; ah = (ah + 1) % 52; ac--;
        fd.push({ player: "A", card: c });
      }
      const bf = Math.min(3, bc);
      for (let f = 0; f < bf; f++) {
        const c = cards[52 + (bh % 52)]; pot[pc++] = c; bh = (bh + 1) % 52; bc--;
        fd.push({ player: "B", card: c });
      }
      warCards.push({ depth: warDepth, faceDown: fd });

      if (ac === 0 && bc === 0) { gameOver = true; break; }
      if (ac === 0) { gameOver = true; break; }
      if (bc === 0) { gameOver = true; break; }

      const na = cards[ah % 52]; ah = (ah + 1) % 52; ac--;
      const nb = cards[52 + (bh % 52)]; bh = (bh + 1) % 52; bc--;
      pot[pc++] = na; pot[pc++] = nb;
      warCards.push({ depth: warDepth, faceUp: { a: na, b: nb } });
      wa = cardRank(na); wb = cardRank(nb);
    }

    const rd = { round: r, cardA: ca, cardB: cb, type: warDepth > 0 ? "war" : "normal", warDepth, warCards, potSize: pc, aCount: ac, bCount: bc };

    if (gameOver) {
      if (ac === 0 && bc === 0) rd.roundWinner = "draw";
      else if (ac === 0) rd.roundWinner = "B";
      else rd.roundWinner = "A";
      rd.type = "war-end"; rounds.push(rd); break;
    }

    // Shuffle pot (matches GPU)
    for (let si = pc - 1; si > 0; si--) { const sj = pcg(state) % (si + 1); const tmp = pot[si]; pot[si] = pot[sj]; pot[sj] = tmp; }

    if (wa > wb) {
      for (let p = 0; p < pc; p++) { cards[at_ % 52] = pot[p]; at_ = (at_ + 1) % 52; ac++; }
      rd.roundWinner = "A";
    } else if (wb > wa) {
      for (let p = 0; p < pc; p++) { cards[52 + (bt_ % 52)] = pot[p]; bt_ = (bt_ + 1) % 52; bc++; }
      rd.roundWinner = "B";
    }
    rd.aCountAfter = ac; rd.bCountAfter = bc;
    rounds.push(rd);
    if (r === 19999) rounds.push({ round: r + 1, type: "stalemate", aCount: ac, bCount: bc });
  }
  return { seed, rounds };
}

// ═══════════════════════════════════════════════════════════════════
// GPU
// ═══════════════════════════════════════════════════════════════════
const FIELDS = 7;
// 65536 is the sweet spot: big enough to amortize dispatch overhead,
// small enough that JS result processing doesn't block the main thread.
let BATCH = 65536;

async function initGPU() {
  if (!navigator.gpu) throw new Error("WebGPU not supported");
  const ad = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!ad) throw new Error("No GPU adapter");
  const dev = await ad.requestDevice();
  let nm = "";
  try { const i = await ad.requestAdapterInfo(); nm = i.device || i.description || ""; } catch (e) {}
  return { dev, nm };
}

function makePipe(dev) {
  return dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: SHADER }), entryPoint: "main" } });
}

async function runBatch(dev, pipe, seedStart, count) {
  const n = count || BATCH;
  const resBytes = n * FIELDS * 4;
  const sd = new Uint32Array(n);
  for (let i = 0; i < n; i++) sd[i] = (seedStart + i) >>> 0;

  const sb = dev.createBuffer({ size: sd.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(sb, 0, sd);
  const rb = dev.createBuffer({ size: resBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const mb = dev.createBuffer({ size: resBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: sb } }, { binding: 1, resource: { buffer: rb } }] });
  const e = dev.createCommandEncoder(), p = e.beginComputePass();
  p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(Math.ceil(n / 64)); p.end();
  e.copyBufferToBuffer(rb, 0, mb, 0, resBytes);
  dev.queue.submit([e.finish()]);
  await mb.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(mb.getMappedRange().slice(0));
  mb.unmap(); sb.destroy(); rb.destroy(); mb.destroy();
  return { data: out, count: n };
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK — tests multiple batch sizes to find optimal
// ═══════════════════════════════════════════════════════════════════
async function runBenchmark(dev, pipe) {
  const N_CPU = 4096;

  // Warm up GPU
  await runBatch(dev, pipe, 999999, 1024);

  // CPU benchmark
  const cpuStart = performance.now();
  for (let i = 0; i < N_CPU; i++) cpuSimGame(i + 1);
  const cpuMs = performance.now() - cpuStart;
  const cpuRate = (N_CPU / cpuMs) * 1000;

  // GPU benchmark — test batch sizes up to 65536
  const sizes = [1024, 4096, 16384, 65536];
  const results = [];
  for (const sz of sizes) {
    let best = Infinity;
    for (let run = 0; run < 2; run++) {
      const t = performance.now();
      await runBatch(dev, pipe, 700000 + run * sz, sz);
      best = Math.min(best, performance.now() - t);
    }
    results.push({ size: sz, ms: best, rate: Math.round((sz / best) * 1000) });
  }

  // Pick best performing size
  results.sort((a, b) => b.rate - a.rate);
  const best = results[0];
  BATCH = best.size;

  return {
    cpuRate: Math.round(cpuRate),
    gpuRate: best.rate,
    speedup: best.rate / cpuRate,
    cpuN: N_CPU,
    gpuN: best.size,
    cpuMs: cpuMs.toFixed(1),
    gpuMs: best.ms.toFixed(1),
    bestBatch: best.size,
    allResults: results,
  };
}

// ═══════════════════════════════════════════════════════════════════
// SMOOTH VALUE
// ═══════════════════════════════════════════════════════════════════
class SmoothVal {
  constructor(v = 0) { this.display = v; this.target = v; this.rate = 0; }
  set(target, rate) { this.target = target; this.rate = rate; }
  snap(v) { this.display = v; this.target = v; }
  tick(dt) {
    if (this.rate > 0 && this.display < this.target) {
      this.display = Math.min(this.target, this.display + this.rate * dt);
    } else { this.display = this.target; }
    return this.display;
  }
}

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS & HELPERS
// ═══════════════════════════════════════════════════════════════════
const SECS_PER_ROUND = 3.5;
const SECS_PER_WAR = 12;
const SECS_PER_YEAR = 365.25 * 24 * 3600;

function fmt(n) { if (n >= 1e9) return (n / 1e9).toFixed(2) + "B"; if (n >= 1e6) return (n / 1e6).toFixed(2) + "M"; if (n >= 1e3) return (n / 1e3).toFixed(1) + "K"; return n.toLocaleString(); }
function fmtX(n) { return n.toLocaleString(); }
const INIT = () => ({ totalGames: 0, aWins: 0, bWins: 0, draws: 0, totalRounds: 0, totalWars: 0, maxRounds: 0, minRounds: 999999, maxWars: 0, maxStreak: 0, biggestWar: 0, histo: new Array(50).fill(0) });

// ═══════════════════════════════════════════════════════════════════
// REPLAY VIEWER
// ═══════════════════════════════════════════════════════════════════
function ReplayViewer({ outlier, onClose }) {
  const [replay, setReplay] = useState(null);
  const [vr, setVr] = useState(0);
  const [playing, setPlaying] = useState(false);
  const tmr = useRef(null);
  useEffect(() => { if (outlier?.seed !== undefined) { setReplay(replayGame(outlier.seed)); setVr(0); } return () => { if (tmr.current) clearInterval(tmr.current); }; }, [outlier]);
  useEffect(() => {
    if (playing && replay) { tmr.current = setInterval(() => { setVr(v => { if (v >= replay.rounds.length - 1) { setPlaying(false); return v; } return v + 1; }); }, 100); }
    else { if (tmr.current) clearInterval(tmr.current); }
    return () => { if (tmr.current) clearInterval(tmr.current); };
  }, [playing, replay]);
  if (!replay) return null;
  const rd = replay.rounds[vr];
  if (!rd) return null;
  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.88)", zIndex: 999, display: "flex", alignItems: "center", justifyContent: "center", padding: 12 }}>
      <div style={{ background: "#13151a", border: "1px solid #2a2d35", borderRadius: 14, padding: 18, maxWidth: 460, width: "100%", maxHeight: "88vh", overflow: "auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div>
            <div style={{ color: "#e8c44a", fontSize: 14, fontWeight: 700 }}>Game Replay</div>
            <div style={{ color: "#4a4f5a", fontSize: 10, fontFamily: "monospace" }}>seed: {outlier.seed} &middot; {replay.rounds.length} rounds</div>
            <div style={{ color: "#3a3f4a", fontSize: 9, marginTop: 2 }}>Simulation paused during replay</div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "1px solid #2a2d35", color: "#6a6f7a", borderRadius: 6, padding: "5px 12px", cursor: "pointer", fontSize: 12 }}>&times; Close</button>
        </div>
        <div style={{ display: "flex", height: 8, borderRadius: 4, overflow: "hidden", marginBottom: 6, background: "#0a0c10" }}>
          <div style={{ width: `${((rd.aCountAfter ?? rd.aCount ?? 26) / 52) * 100}%`, background: "#63b6ff", transition: "width 0.12s" }} />
          <div style={{ flex: 1, background: "#ff7eb3", transition: "width 0.12s" }} />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#5a5f6a", marginBottom: 14 }}>
          <span>A: {rd.aCountAfter ?? rd.aCount ?? "?"}</span><span>B: {rd.bCountAfter ?? rd.bCount ?? "?"}</span>
        </div>
        <div style={{ textAlign: "center", marginBottom: 16 }}>
          <div style={{ color: "#5a5f6a", fontSize: 10, letterSpacing: 1, marginBottom: 8 }}>ROUND {rd.round + 1}</div>
          {(rd.type === "end" || rd.type === "stalemate") ? (
            <div style={{ color: rd.type === "stalemate" ? "#6a6f7a" : "#e8c44a", fontSize: 18, fontWeight: 700 }}>{rd.type === "stalemate" ? "STALEMATE" : `PLAYER ${rd.winner} WINS`}</div>
          ) : (
            <div style={{ display: "flex", justifyContent: "center", gap: 20, alignItems: "center" }}>
              <CCard card={rd.cardA} label="A" win={rd.roundWinner === "A"} />
              <div style={{ color: rd.type.startsWith("war") ? "#ff6b6b" : "#3a3f4a", fontSize: 14, fontWeight: 700 }}>{rd.type.startsWith("war") ? "\u2694\ufe0f" + (rd.warDepth > 1 ? " x" + rd.warDepth : "") : "vs"}</div>
              <CCard card={rd.cardB} label="B" win={rd.roundWinner === "B"} />
            </div>
          )}
          {rd.potSize > 2 && <div style={{ color: "#a78bfa", fontSize: 11, marginTop: 8 }}>Pot: {rd.potSize} cards</div>}
        </div>
        <input type="range" min={0} max={replay.rounds.length - 1} value={vr} onChange={e => { setVr(Number(e.target.value)); setPlaying(false); }} style={{ width: "100%", accentColor: "#63b6ff", marginBottom: 8 }} />
        <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 14 }}>
          <RB onClick={() => { setVr(0); setPlaying(false); }}>&laquo;</RB>
          <RB onClick={() => setVr(v => Math.max(0, v - 1))}>&lsaquo;</RB>
          <RB onClick={() => setPlaying(p => !p)} accent>{playing ? <span>&#9646;&#9646;</span> : <span>&#9654;</span>}</RB>
          <RB onClick={() => setVr(v => Math.min(replay.rounds.length - 1, v + 1))}>&rsaquo;</RB>
          <RB onClick={() => { setVr(replay.rounds.length - 1); setPlaying(false); }}>&raquo;</RB>
        </div>
        <div style={{ maxHeight: 110, overflow: "auto", borderTop: "1px solid #1a1d24", paddingTop: 8 }}>
          {replay.rounds.slice(Math.max(0, vr - 5), vr + 1).map(r => (
            <div key={r.round} style={{ fontSize: 10, color: r.round === vr ? "#c8ccd4" : "#3a3f4a", padding: "1px 0", fontFamily: "monospace" }}>
              R{String(r.round + 1).padStart(4, "\u2007")}: {r.type === "end" ? `${r.winner} wins!` : r.type === "stalemate" ? "stalemate" :
              `${cardName(r.cardA)} vs ${cardName(r.cardB)} → ${r.roundWinner}${r.warDepth > 0 ? " (war×" + r.warDepth + ")" : ""}`}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function CCard({ card, label, win }) {
  const red = card % 4 === 1 || card % 4 === 2;
  return (
    <div style={{ background: win ? "#122214" : "#0d0f13", border: `2px solid ${win ? "#4ade80" : "#2a2d35"}`, borderRadius: 10, padding: "8px 14px", minWidth: 58, textAlign: "center" }}>
      <div style={{ fontSize: 9, color: "#5a5f6a", marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 800, color: red ? "#ff6b6b" : "#e0e0e0", fontFamily: "monospace" }}>{cardName(card)}</div>
    </div>
  );
}

function RB({ onClick, children, accent }) {
  return (
    <button onClick={onClick} style={{ background: accent ? "#63b6ff" : "#1a1d24", color: accent ? "#0d0f13" : "#8a8f98", border: "1px solid #2a2d35", borderRadius: 6, padding: "6px 12px", cursor: "pointer", fontSize: 16, fontWeight: 700, minWidth: 38, lineHeight: 1 }}>{children}</button>
  );
}

// ═══════════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════════
export default function WarSim() {
  const [status, setStatus] = useState("initializing");
  const [error, setError] = useState(null);
  const [gpuName, setGpuName] = useState("");
  const [stats, setStats] = useState(null);
  const [outliers, setOutliers] = useState([]);
  const [rate, setRate] = useState(0);
  const [running, setRunning] = useState(false);
  const [viewOutlier, setViewOutlier] = useState(null);

  const sm = useRef({
    games: new SmoothVal(), rounds: new SmoothVal(), wars: new SmoothVal(),
    aWins: new SmoothVal(), bWins: new SmoothVal(), draws: new SmoothVal(),
  });
  const refs = useRef({});
  const setRef = useCallback((key) => (el) => { refs.current[key] = el; }, []);
  const gpu = useRef(null), pipe = useRef(null), sr = useRef(INIT()), or_ = useRef([]);
  const go = useRef(false), ready = useRef(false), curRate = useRef(0);
  const seedCounter = useRef(0);
  const baseSeed = useRef(0);
  const wasRunning = useRef(false);

  // Auto-pause sim when replay opens, resume when it closes
  const openReplay = useCallback((outlier) => {
    wasRunning.current = go.current;
    if (go.current) go.current = false;
    setViewOutlier(outlier);
  }, []);
  const closeReplay = useCallback(() => {
    setViewOutlier(null);
    // Resume will be triggered by user pressing Resume
  }, []);

  // ── Animation loop ──
  useEffect(() => {
    let raf, prev = performance.now();
    const tick = (now) => {
      const dt = (now - prev) / 1000; prev = now;
      const v = sm.current;
      const g = v.games.tick(dt), r = v.rounds.tick(dt), w = v.wars.tick(dt);
      const a = v.aWins.tick(dt), b = v.bWins.tick(dt), d = v.draws.tick(dt);
      const R = refs.current;
      if (R.games) R.games.textContent = Math.floor(g).toLocaleString();
      if (R.rate) R.rate.textContent = curRate.current > 0 ? fmt(Math.round(curRate.current)) + " games/sec" : "";
      if (g > 0) {
        if (R.pA) R.pA.textContent = ((a / g) * 100).toFixed(3) + "%";
        if (R.pAsub) R.pAsub.textContent = fmtX(Math.floor(a));
        if (R.pB) R.pB.textContent = ((b / g) * 100).toFixed(3) + "%";
        if (R.pBsub) R.pBsub.textContent = fmtX(Math.floor(b));
        if (R.pD) R.pD.textContent = ((d / g) * 100).toFixed(4) + "%";
        if (R.pDsub) R.pDsub.textContent = fmtX(Math.floor(d));
        if (R.avgR) R.avgR.textContent = (r / g).toFixed(1);
        if (R.avgW) R.avgW.textContent = (w / g).toFixed(2);
        if (R.totW) R.totW.textContent = fmt(Math.floor(w));
        const secs = (r - w) * SECS_PER_ROUND + w * SECS_PER_WAR;
        const years = secs / SECS_PER_YEAR;
        if (R.years) R.years.textContent = years >= 100 ? fmtX(Math.round(years)) : years.toFixed(1);
        const lifetimes = secs / (72 * 365.25 * 16 * 3600);
        if (R.lives) R.lives.textContent = lifetimes >= 1 ? (lifetimes >= 1000 ? fmtX(Math.round(lifetimes)) : lifetimes.toFixed(1)) + " human " + (lifetimes >= 2 ? "lifetimes" : "lifetime") : "";
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // ── Load saved + init GPU + benchmark ──
  useEffect(() => {
    (async () => {
      // Load saved
      try {
        const saved = await window.storage.get("war-v6");
        if (saved) {
          const d = JSON.parse(saved.value);
          sr.current = { ...INIT(), ...d.s };
          or_.current = d.o || [];
          baseSeed.current = d.baseSeed || 0;
          seedCounter.current = d.seedCounter || sr.current.totalGames;
          const s = sr.current;
          sm.current.games.snap(s.totalGames); sm.current.rounds.snap(s.totalRounds); sm.current.wars.snap(s.totalWars);
          sm.current.aWins.snap(s.aWins); sm.current.bWins.snap(s.bWins); sm.current.draws.snap(s.draws);
          setStats({ ...s }); setOutliers([...or_.current]);
        } else {
          // First run: pick a random base seed
          baseSeed.current = (Math.random() * 0xFFFFFFFF) >>> 0;
        }
      } catch (e) {
        baseSeed.current = (Math.random() * 0xFFFFFFFF) >>> 0;
      }
      ready.current = true;

      // Init GPU
      try {
        const { dev, nm } = await initGPU();
        gpu.current = dev; pipe.current = makePipe(dev); setGpuName(nm);
        // Auto-tune batch size
        await runBenchmark(dev, pipe.current);
        setStatus("ready");
      } catch (e) { setError(e.message); setStatus("error"); }
    })();
  }, []);

  const save = useCallback(async () => {
    if (!ready.current) return;
    try { await window.storage.set("war-v6", JSON.stringify({ s: sr.current, o: or_.current, baseSeed: baseSeed.current, seedCounter: seedCounter.current })); } catch (e) {}
  }, []);

  const loop = useCallback(async () => {
    if (!gpu.current || !pipe.current) return;
    go.current = true; setRunning(true);
    let saves = 0, t0 = performance.now(), gw = 0;
    const dev = gpu.current, pi = pipe.current;
    const batchSize = BATCH;

    // Double-buffer: submit next GPU dispatch while processing current results
    let seedStart = (baseSeed.current + seedCounter.current) >>> 0;
    seedCounter.current += batchSize;
    let pending = runBatch(dev, pi, seedStart, batchSize);

    while (go.current) {
      try {
        const { data, count } = await pending;

        // Immediately submit next dispatch (GPU works while we process)
        if (go.current) {
          seedStart = (baseSeed.current + seedCounter.current) >>> 0;
          seedCounter.current += batchSize;
          pending = runBatch(dev, pi, seedStart, batchSize);
        }

        // ── Hot path: process results with minimal allocation ──
        const s = sr.current;
        for (let i = 0; i < count; i++) {
          const o = i * FIELDS;
          const win = data[o], tr = data[o + 1], wc = data[o + 2], ms = data[o + 3], bw = data[o + 4], seed = data[o + 5];
          s.totalGames++;
          if (win === 0) s.aWins++; else if (win === 1) s.bWins++; else s.draws++;
          s.totalRounds += tr; s.totalWars += wc;
          s.histo[Math.min(49, tr / 100 | 0)]++;

          // Only check outliers if this game beats a record or is a draw (rare)
          if (tr > s.maxRounds || (tr > 0 && tr < s.minRounds) || wc > s.maxWars || ms > s.maxStreak || bw > s.biggestWar || win === 2) {
            let reasons = "";
            if (win === 2) { reasons += "Draw"; }
            if (tr > s.maxRounds) { s.maxRounds = tr; reasons += (reasons ? " | " : "") + "Longest game: " + tr + " rnds"; }
            if (tr > 0 && tr < s.minRounds) { s.minRounds = tr; reasons += (reasons ? " | " : "") + "Shortest game: " + tr + " rnds"; }
            if (wc > s.maxWars) { s.maxWars = wc; reasons += (reasons ? " | " : "") + "Most wars: " + wc; }
            if (ms > s.maxStreak) { s.maxStreak = ms; reasons += (reasons ? " | " : "") + "Longest streak: " + ms; }
            if (bw > s.biggestWar) { s.biggestWar = bw; reasons += (reasons ? " | " : "") + "Deepest war: " + bw + " deep"; }
            or_.current = [{ win, tr, wc, ms, bw, seed, reason: reasons, gn: s.totalGames, ts: Date.now() }, ...or_.current.slice(0, 49)];
          }
        }

        gw += count; saves++;
        const now = performance.now();
        if (now - t0 > 400) {
          const r = gw / ((now - t0) / 1000); curRate.current = r;
          const v = sm.current, avgRpg = s.totalRounds / s.totalGames, avgWpg = s.totalWars / s.totalGames;
          v.games.set(s.totalGames, r); v.rounds.set(s.totalRounds, avgRpg * r); v.wars.set(s.totalWars, avgWpg * r);
          v.aWins.set(s.aWins, (s.aWins / s.totalGames) * r); v.bWins.set(s.bWins, (s.bWins / s.totalGames) * r); v.draws.set(s.draws, (s.draws / s.totalGames) * r);
          setRate(r); setStats({ ...s }); setOutliers([...or_.current]);
          t0 = now; gw = 0;
        }
        if (saves >= 25) { save(); saves = 0; }
      } catch (e) { console.error("GPU:", e); setError("GPU: " + e.message); go.current = false; break; }
      await new Promise(r => setTimeout(r, 0));
    }
    setRunning(false); curRate.current = 0; save();
  }, [save]);

  const stop = useCallback(() => { go.current = false; }, []);
  const reset = useCallback(async () => {
    go.current = false; sr.current = INIT(); or_.current = [];
    baseSeed.current = (Math.random() * 0xFFFFFFFF) >>> 0;
    seedCounter.current = 0;
    const v = sm.current; v.games.snap(0); v.rounds.snap(0); v.wars.snap(0); v.aWins.snap(0); v.bWins.snap(0); v.draws.snap(0);
    curRate.current = 0; setStats(null); setOutliers([]); setRate(0); setError(null);
    try { await window.storage.delete("war-v6"); } catch (e) {}
  }, []);

  if (status === "error" && !stats) {
    return (
      <div style={P.pg}><div style={{ background: "#13151a", border: "1px solid #2a1a1a", borderRadius: 12, padding: 28, textAlign: "center", marginTop: 40 }}>
        <div style={{ fontSize: 28, marginBottom: 8 }}>&#9888;</div>
        <div style={{ fontSize: 16, fontWeight: 700, color: "#ff6b6b" }}>WebGPU Not Available</div>
        <div style={{ color: "#8a8f98", marginTop: 8, fontSize: 13 }}>{error}</div>
      </div></div>
    );
  }

  const s = stats, ag = s && s.totalGames > 0;
  const hM = s ? Math.max(...s.histo, 1) : 1;

  return (
    <div style={P.pg}>
      {viewOutlier && <ReplayViewer outlier={viewOutlier} onClose={closeReplay} />}

      <div style={P.hd}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <span style={{ fontSize: 26, fontWeight: 800, color: "#fff", letterSpacing: 4 }}>WAR</span>
          <span style={{ fontSize: 11, color: "#63b6ff", letterSpacing: 3, fontWeight: 600 }}>GPU SIMULATOR</span>
        </div>
        <div style={{ fontSize: 11, color: "#4a4f5a", marginTop: 4 }}>
          {fmtX(BATCH)}/dispatch &middot; {gpuName || "WebGPU"} &middot; seed {baseSeed.current}
        </div>
      </div>

      <div style={P.ct}>
        <div style={{ fontSize: 10, color: "#5a5f6a", letterSpacing: 2, textTransform: "uppercase", marginBottom: 6 }}>Total Games Simulated</div>
        <span ref={setRef("games")} style={{ fontSize: 34, fontWeight: 800, color: "#fff", fontFamily: "'Courier New',monospace", letterSpacing: 1 }}>0</span>
        <div ref={setRef("rate")} style={{ fontSize: 13, color: "#63b6ff", marginTop: 4, fontFamily: "monospace", minHeight: 18 }}></div>
        {error && <div style={{ fontSize: 11, color: "#ff6b6b", marginTop: 6 }}>{error}</div>}
        <div style={{ display: "flex", gap: 8, marginTop: 14, justifyContent: "center" }}>
          {!running && <button onClick={loop} disabled={status === "initializing"} style={P.gob}>{status === "initializing" ? "Benchmarking…" : ag ? "▶  RESUME" : "▶  START"}</button>}
          {running && <button onClick={stop} style={P.stb}>■  PAUSE</button>}
          {ag && !running && <button onClick={reset} style={P.rsb}>RESET</button>}
        </div>
      </div>

      {ag && <>
        <div style={{ ...P.bx, textAlign: "center" }}>
          <div style={{ fontSize: 9, color: "#5a5f6a", letterSpacing: 2, textTransform: "uppercase", marginBottom: 6 }}>Equivalent human playtime</div>
          <div style={{ fontFamily: "monospace", fontSize: 22, color: "#4ade80", fontWeight: 700 }}>
            <span ref={setRef("years")}></span> <span style={{ fontSize: 13, color: "#6a6f7a", fontWeight: 500 }}>years</span>
          </div>
          <div ref={setRef("lives")} style={{ fontSize: 10, color: "#5a5f6a", marginTop: 4, minHeight: 14 }}></div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6, marginBottom: 6 }}>
          <div style={P.sc}><div style={P.scl}>Player A</div><div ref={setRef("pA")} style={{ ...P.scv, color: "#63b6ff" }}></div><div ref={setRef("pAsub")} style={P.scs}></div></div>
          <div style={P.sc}><div style={P.scl}>Player B</div><div ref={setRef("pB")} style={{ ...P.scv, color: "#ff7eb3" }}></div><div ref={setRef("pBsub")} style={P.scs}></div></div>
          <div style={P.sc}><div style={P.scl}>Draws</div><div ref={setRef("pD")} style={{ ...P.scv, color: "#6a6f7a" }}></div><div ref={setRef("pDsub")} style={P.scs}></div></div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6, marginBottom: 12 }}>
          <div style={P.sc}><div style={P.scl}>Avg Rounds</div><div ref={setRef("avgR")} style={{ ...P.scv, color: "#e8c44a" }}></div></div>
          <div style={P.sc}><div style={P.scl}>Avg Wars</div><div ref={setRef("avgW")} style={{ ...P.scv, color: "#a78bfa" }}></div></div>
          <div style={P.sc}><div style={P.scl}>Total Wars</div><div ref={setRef("totW")} style={{ ...P.scv, color: "#a78bfa" }}></div></div>
        </div>

        <div style={P.bx}>
          <div style={P.sl}>All-Time Records</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
            <RC i="&#128207;" l="Longest" v={fmtX(s.maxRounds) + " rnds"} />
            <RC i="&#9889;" l="Shortest" v={s.minRounds >= 999999 ? "—" : s.minRounds + " rnds"} />
            <RC i="&#9876;" l="Most Wars" v={fmtX(s.maxWars)} />
            <RC i="&#128293;" l="Best Streak" v={fmtX(s.maxStreak)} />
            <RC i="&#128165;" l="Deepest War" v={s.biggestWar + " deep"} />
          </div>
        </div>

        <div style={P.bx}>
          <div style={P.sl}>Game Length Distribution</div>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 1, height: 70 }}>
            {s.histo.map((v, i) => (
              <div key={i} title={i * 100 + "–" + (i * 100 + 99) + ": " + fmtX(v)} style={{ flex: 1, minWidth: 2, borderRadius: "2px 2px 0 0", height: v > 0 ? Math.max(2, (v / hM) * 100) + "%" : "0%", background: v === hM ? "#e8c44a" : "rgba(99,182,255," + (0.25 + 0.75 * (v / hM)) + ")", transition: "height 0.3s" }} />
            ))}
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, fontSize: 9, color: "#4a4f5a" }}>
            <span>0</span><span>1K</span><span>2K</span><span>3K</span><span>4K</span><span>5K+</span>
          </div>
        </div>

        {outliers.length > 0 && <div style={P.bx}>
          <div style={P.sl}>Record-Breaking Games <span style={{ color: "#3a3f4a", fontWeight: 400, textTransform: "none" }}>tap to replay</span></div>
          {outliers.slice(0, 15).map((o, i) => (
            <div key={i} onClick={() => openReplay(o)} style={{ background: "#0a0c10", border: "1px solid #1a1d24", borderRadius: 6, padding: "7px 10px", marginBottom: 4, cursor: "pointer" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <span style={{ color: "#e8c44a", fontWeight: 600, fontSize: 12 }}>{o.reason}</span>
                <span style={{ color: "#3a3f4a", fontSize: 9 }}>#{fmtX(o.gn)}</span>
              </div>
              <div style={{ color: "#4a4f5a", fontSize: 10, marginTop: 2 }}>
                {o.tr} rnds &middot; {o.wc} wars &middot; chain:{o.bw} &middot; {o.win === 0 ? "A wins" : o.win === 1 ? "B wins" : "draw"}
                <span style={{ color: "#2a2d35", marginLeft: 6 }}>seed:{o.seed}</span>
              </div>
            </div>
          ))}
        </div>}
      </>}
    </div>
  );
}

function RC({ i, l, v }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "3px 0" }}>
      <span style={{ fontSize: 14 }} dangerouslySetInnerHTML={{ __html: i }} />
      <div>
        <div style={{ color: "#6a6f7a", fontSize: 9, textTransform: "uppercase", letterSpacing: 0.5 }}>{l}</div>
        <div style={{ color: "#e0e0e0", fontSize: 13, fontWeight: 600, fontFamily: "monospace" }}>{v}</div>
      </div>
    </div>
  );
}

const P = {
  pg: { minHeight: "100vh", background: "#0d0f13", color: "#e0e0e0", fontFamily: "system-ui,sans-serif", padding: "20px 14px", maxWidth: 540, margin: "0 auto" },
  hd: { marginBottom: 18, borderBottom: "1px solid #1a1d24", paddingBottom: 12 },
  ct: { background: "#13151a", border: "1px solid #1e2028", borderRadius: 12, padding: "20px 16px", textAlign: "center", marginBottom: 14 },
  gob: { flex: 1, padding: "11px 20px", background: "#63b6ff", color: "#0d0f13", border: "none", borderRadius: 8, fontSize: 13, fontWeight: 700, cursor: "pointer", letterSpacing: 1 },
  stb: { flex: 1, padding: "11px 20px", background: "#ff6b6b", color: "#fff", border: "none", borderRadius: 8, fontSize: 13, fontWeight: 700, cursor: "pointer", letterSpacing: 1 },
  rsb: { padding: "11px 14px", background: "transparent", color: "#5a5f6a", border: "1px solid #2a2d35", borderRadius: 8, fontSize: 11, cursor: "pointer" },
  bx: { background: "#13151a", border: "1px solid #1e2028", borderRadius: 10, padding: 14, marginBottom: 12 },
  sl: { color: "#6a6f7a", fontSize: 10, marginBottom: 8, letterSpacing: 1.5, textTransform: "uppercase" },
  sc: { background: "#13151a", border: "1px solid #1e2028", borderRadius: 8, padding: "10px 8px", textAlign: "center" },
  scl: { color: "#5a5f6a", fontSize: 9, textTransform: "uppercase", letterSpacing: 1, marginBottom: 3 },
  scv: { fontSize: 20, fontWeight: 700, fontFamily: "monospace" },
  scs: { color: "#3a3f4a", fontSize: 10, marginTop: 1 },
};
