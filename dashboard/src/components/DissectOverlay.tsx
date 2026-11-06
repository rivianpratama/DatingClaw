"use client";

import React from "react";

export interface BoxData {
  has_face: boolean;
  face_box: [number, number, number, number]; // [x1, y1, x2, y2] normalized 0..1
  photo_box: [number, number, number, number]; // [x1, y1, x2, y2]
  ctx_box: [number, number, number, number];
  kps: [number, number][]; // 5 facial landmarks
  det_score: number;
}

interface DissectOverlayProps {
  boxData?: BoxData | null;
  v6r5Score?: number | null;
  loraScore?: number | null;
  finalScore?: number | null;
  action?: string | null;
  variant?: "thumbnail" | "modal" | "live";
  showFace?: boolean;
  showContext?: boolean;
  showLandmarks?: boolean;
  className?: string;
}

export default function DissectOverlay({
  boxData,
  v6r5Score,
  loraScore,
  finalScore,
  action,
  variant = "thumbnail",
  showFace = true,
  showContext = true,
  showLandmarks = true,
  className = "",
}: DissectOverlayProps) {
  if (!boxData) return null;

  const isModal = variant === "modal";
  const isLive = variant === "live";
  const hasFace = boxData.has_face;
  const isRight = action === "right";

  const faceBox = boxData.face_box || [0.32, 0.20, 0.68, 0.44];
  const photoBox = boxData.photo_box || [0.0, 0.078, 1.0, 0.60];
  const kps = boxData.kps || [];

  const fx = faceBox[0] * 100;
  const fy = faceBox[1] * 100;
  const fw = (faceBox[2] - faceBox[0]) * 100;
  const fh = (faceBox[3] - faceBox[1]) * 100;

  const px = photoBox[0] * 100;
  const py = photoBox[1] * 100;
  const pw = (photoBox[2] - photoBox[0]) * 100;
  const ph = (photoBox[3] - photoBox[1]) * 100;

  const fmtScore = (s: number | null | undefined, weight: string) => {
    if (s === null || s === undefined) return weight;
    return `${s >= 0 ? "+" : ""}${s.toFixed(2)}z`;
  };

  return (
    <div className={`absolute inset-0 pointer-events-none select-none overflow-hidden z-20 ${className}`}>
      {/* Subtle Matrix Ambient Framing */}
      <div className="absolute inset-0 ring-1 ring-inset ring-white/10" />

      {/* ─── 1. Photo Context Boundary (SigLIP SO400M - Amber Accent) ─── */}
      {showContext && (
        <div
          className="absolute border border-dashed border-amber-400/80 shadow-[0_0_12px_rgba(245,158,11,0.25)] transition-all duration-300 pointer-events-none"
          style={{
            left: `${px}%`,
            top: `${py}%`,
            width: `${pw}%`,
            height: `${ph}%`,
          }}
        >
          {/* Top-Right Badge: SigLIP SO400M */}
          <div className="absolute top-0 right-0 bg-[#07090e]/90 border-l border-b border-amber-500/60 text-amber-300 font-mono font-bold px-1.5 py-0.5 rounded-bl shadow-md flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
            <span className={isModal ? "text-[10px]" : "text-[8px]"}>
              SIGLIP CONTEXT [70%] {fmtScore(loraScore, "70%")}
            </span>
          </div>

          {/* Photo boundary corner accents */}
          <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-amber-400" />
          <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-amber-400" />
          <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-amber-400" />
        </div>
      )}

      {/* ─── 2. Facial Biometrics Reticle (ArcFace 512D + DINOv3 - Electric Cyan / Sky Accent) ─── */}
      {showFace && hasFace && (
        <div
          className="absolute border border-cyan-400/90 shadow-[0_0_15px_rgba(6,182,212,0.35)] transition-all duration-300 pointer-events-none"
          style={{
            left: `${fx}%`,
            top: `${fy}%`,
            width: `${fw}%`,
            height: `${fh}%`,
          }}
        >
          {/* L-Shaped Corner Brackets */}
          <div className="absolute -top-1 -left-1 w-2.5 h-2.5 border-t-2 border-l-2 border-cyan-300" />
          <div className="absolute -top-1 -right-1 w-2.5 h-2.5 border-t-2 border-r-2 border-cyan-300" />
          <div className="absolute -bottom-1 -left-1 w-2.5 h-2.5 border-b-2 border-l-2 border-cyan-300" />
          <div className="absolute -bottom-1 -right-1 w-2.5 h-2.5 border-b-2 border-r-2 border-cyan-300" />

          {/* Center Crosshair (Modal & Live) */}
          {(isModal || isLive) && (
            <div className="absolute inset-0 flex items-center justify-center opacity-70">
              <span className="text-cyan-300 text-xs font-mono leading-none font-bold">+</span>
            </div>
          )}

          {/* Face Biometric Chip Tag */}
          <div
            className={`absolute -top-4 left-0 bg-[#07090e]/95 border border-cyan-500/80 text-cyan-200 font-mono font-bold px-1.5 py-0.2 rounded shadow-md whitespace-nowrap flex items-center gap-1 ${
              isModal ? "text-[9px] -top-5" : "text-[7.5px]"
            }`}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            <span>FACE BIOMETRIC [30%] {fmtScore(v6r5Score, "30%")}</span>
          </div>

          {/* Micro Coordinates (Modal & Live) */}
          {isModal && (
            <div className="absolute -bottom-4 right-0 bg-[#07090e]/95 border border-cyan-500/50 text-cyan-400/90 text-[8px] font-mono px-1 py-0.2 rounded shadow whitespace-nowrap">
              LOC: [{fx.toFixed(1)}%, {fy.toFixed(1)}%] DIM: [{fw.toFixed(1)}%x{fh.toFixed(1)}%]
            </div>
          )}
        </div>
      )}

      {/* Warning if no face resolved */}
      {showFace && !hasFace && isModal && (
        <div className="absolute top-1/4 inset-x-6 bg-[#0c1017]/95 border border-amber-500/50 rounded-xl p-3 text-center text-amber-300 font-mono text-xs shadow-xl">
          ⚠️ FACE NOT CONFIRMED IN PHOTO REGION
        </div>
      )}

      {/* ─── 3. 5-Point Facial Landmarks (Delicate Cyan / Emerald Keypoints) ─── */}
      {showLandmarks && hasFace && kps.length > 0 && (
        <>
          {kps.map(([kx, ky], idx) => (
            <div
              key={idx}
              className="absolute w-1.5 h-1.5 rounded-full bg-cyan-300 shadow-[0_0_8px_rgba(6,182,212,0.9)] -translate-x-1/2 -translate-y-1/2 pointer-events-none ring-1 ring-white/60"
              style={{
                left: `${kx * 100}%`,
                top: `${ky * 100}%`,
              }}
              title={`Landmark ${idx + 1}`}
            />
          ))}
        </>
      )}

      {/* ─── 4. Top Telemetry Header Bar (For Modal and Live Intercept) ─── */}
      {(isModal || isLive) && (
        <div className="absolute top-0 inset-x-0 bg-[#07090e]/92 backdrop-blur-md border-b border-white/10 px-3 py-1.5 flex items-center justify-between text-white font-mono z-30">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                isRight ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" : "bg-rose-400 shadow-[0_0_8px_rgba(251,113,133,0.8)]"
              }`}
            />
            <span className="text-[10px] tracking-wider uppercase font-bold text-gray-300">
              SIGNAL CV:{" "}
              <b className={isRight ? "text-emerald-400" : "text-rose-400"}>
                {isRight ? "OK (RIGHT)" : "PASS (LEFT)"}
              </b>
            </span>
          </div>

          <div className="flex items-center gap-2 text-[10px]">
            <span className="text-gray-400 font-semibold">FUSED Z:</span>
            <span className="text-amber-300 font-bold">
              {finalScore !== null && finalScore !== undefined
                ? `${finalScore >= 0 ? "+" : ""}${finalScore.toFixed(2)}z`
                : "--"}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
