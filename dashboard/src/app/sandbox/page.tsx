"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";

type AnalysisResult = {
  success: boolean;
  decision: "RIGHT" | "LEFT";
  primary_score: number;
  threshold: number;
  threshold_diff: number;
  model_version: string;
  model_display_name: string;
  formula_description?: string;
  primary_face_weight: number;
  primary_k: number;
  is_preference_mode: boolean;
  base_score: number;
  scores: {
    face_biased: number | null;
    ridge: number | null;
    multimodal: number | null;
    knn: number;
    original_face_biased: number | null;
    chandra_7k_face_biased: number | null;
    round1_face_biased: number | null;
    round2_face_biased: number | null;
    round3_face_biased: number | null;
    bumble_only_face_biased: number | null;
    bumble_only_round2_face_biased: number | null;
    preference_probability: number | null;
    preference_threshold: number | null;
  };
  comparison_text: Record<string, string>;
  details: {
    max_similarity: number;
    mean_similarity: number;
    references_count: number;
  };
  nearest_references: Array<{
    name: string;
    rating: number;
    similarity: number;
  }>;
};

export default function Sandbox() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [modelVersion, setModelVersion] = useState<string>("round3");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Clear preview URL on unmount to avoid memory leaks
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleFileChange = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }
    setImage(file);
    setError(null);
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    setResult(null); // clear previous result
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  };

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const runAnalysis = async (activeModel = modelVersion) => {
    if (!image) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("image", image);
    formData.append("model_version", activeModel);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorJson = await response.json();
        throw new Error(errorJson.error || "Failed to analyze image");
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred during scoring.");
    } finally {
      setLoading(false);
    }
  };

  const handleModelSelect = (version: string) => {
    setModelVersion(version);
    // If an image is already loaded, automatically trigger analysis for the new model selection
    if (image) {
      runAnalysis(version);
    }
  };

  const formatNumber = (num: number | null | undefined) => {
    if (num === null || num === undefined) return "--";
    return num.toFixed(2);
  };

  const selectedBaseCardKey: Record<string, string> = {
    original: "original_face_biased",
    chandra_7k: "chandra_7k_face_biased",
    round1: "round1_face_biased",
    round2: "round2_face_biased",
    round3: "round3_face_biased",
    bumble_only: "bumble_only_face_biased",
    bumble_only_round2: "bumble_only_round2_face_biased",
    experimental1: "round2_face_biased",
    experimental2: "round3_face_biased",
    experimental3: "original_face_biased",
    multimodalx: "round3_face_biased",
    multimodalx2: "round3_face_biased",
    multimodalx3: "round3_face_biased",
    multimodalx4: "round3_face_biased",
    multimodalx5: "round3_face_biased",
    multimodalx6: "round2_face_biased",
  };

  return (
    <main className="dashboard-shell selection:bg-sky-500/30 min-h-screen">
      <div className="dashboard-frame max-w-[1800px] mx-auto">
        
        {/* Header Section */}
        <header className="dashboard-header">
          <div>
            <div className="dashboard-status">
              <span className="dashboard-status-dot" style={{ background: "var(--indigo)", boxShadow: "0 0 0 4px rgba(129, 140, 248, 0.12)" }} />
              Sandbox Environment
            </div>
            <h1 className="dashboard-title">Profile Analyzer Sandbox</h1>
          </div>
          
          <div className="dashboard-actions">
            <Link href="/" className="gallery-link">
              <span className="summary-label">Control Room</span>
              <span className="gallery-title">Live Telemetry</span>
            </Link>
            <Link href="/gallery" className="gallery-link">
              <span className="summary-label">Access Repository</span>
              <span className="gallery-title">View Gallery</span>
            </Link>
          </div>
        </header>

        {/* Dynamic MacBook Pro 13 Viewport Styles */}
        <style dangerouslySetInnerHTML={{ __html: `
          .sandbox-workspace {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.25rem;
            margin-top: 1rem;
          }

          .sandbox-metric-value {
            font-family: var(--font-geist-mono), ui-monospace, SFMono-Regular, Menlo, monospace;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.01em;
          }

          .sandbox-metric-title {
            color: rgba(255, 255, 255, 0.85);
            font-weight: 760;
            line-height: 1.15;
            margin: 0.15rem 0 0.2rem;
          }

          .sandbox-overall-card {
            background: var(--surface-strong);
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
          }

          @media (min-width: 1200px) {
            .sandbox-workspace {
              grid-template-columns: clamp(230px, 16vw, 260px) clamp(320px, 22vw, 340px) minmax(0, 1fr);
              height: calc(100vh - 120px);
              overflow: hidden;
              gap: 1.25rem;
            }

            .sandbox-column-left {
              height: 100%;
              display: flex;
              flex-direction: column;
              gap: 1.25rem;
              justify-content: flex-start;
            }

            .sandbox-column-middle,
            .sandbox-column-right {
              height: 100%;
              overflow-y: auto;
              padding-right: 4px;
              display: flex;
              flex-direction: column;
              gap: 1.25rem;
            }

            .sandbox-column-middle::-webkit-scrollbar,
            .sandbox-column-right::-webkit-scrollbar {
              width: 4px;
            }
            .sandbox-column-middle::-webkit-scrollbar-track,
            .sandbox-column-right::-webkit-scrollbar-track {
              background: transparent;
            }
            .sandbox-column-middle::-webkit-scrollbar-thumb,
            .sandbox-column-right::-webkit-scrollbar-thumb {
              background: rgba(255, 255, 255, 0.08);
              border-radius: 99px;
            }
            .sandbox-column-middle::-webkit-scrollbar-thumb:hover,
            .sandbox-column-right::-webkit-scrollbar-thumb:hover {
              background: rgba(56, 189, 248, 0.25);
              border-radius: 99px;
            }
          }
        ` }} />

        {/* Workspace Layout */}
        <div className="sandbox-workspace">
          
          {/* COLUMN 1: Profile Media Upload */}
          <div className="sandbox-column-left">
            {/* Drag & Drop Visual Uploader Container */}
            <section className="preview-card flex flex-col gap-3">
              <h2 className="section-label">Profile Media Upload</h2>
              
              <div 
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                onClick={triggerFileInput}
                className={`relative w-full aspect-[9/16] rounded-xl border-2 overflow-hidden cursor-pointer transition-all duration-300 ${
                  dragActive 
                    ? "border-sky-400 bg-sky-500/10 scale-[0.99]" 
                    : preview 
                      ? "border-transparent bg-black/40" 
                      : "border-dashed border-white/20 hover:border-white/40 bg-white/[0.02]"
                }`}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
                  className="hidden" 
                  accept="image/*"
                />

                {preview ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img 
                      src={preview} 
                      alt="Uploaded face profile" 
                      className="absolute inset-0 w-full h-full object-cover transition-transform duration-500 hover:scale-105"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/30 pointer-events-none" />
                    
                    <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between text-[10px] text-white/60">
                      <span>Swap image</span>
                      <span className="font-mono truncate max-w-[100px]">{image?.name ? (image.name.length > 12 ? image.name.substring(0, 9) + "..." : image.name) : ""}</span>
                    </div>
                  </>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center p-4 text-center select-none">
                    <svg className="w-8 h-8 text-white/30 mb-3 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span className="text-white/70 font-semibold text-xs">Drag & Drop Image</span>
                    <span className="text-white/40 text-[9px] mt-0.5">or click to browse</span>
                  </div>
                )}

                {/* Loader Overlay */}
                {loading && (
                  <div className="absolute inset-0 bg-black/70 backdrop-blur-sm flex flex-col items-center justify-center z-50">
                    <div className="loading-spinner mb-3" />
                    <span className="text-[9px] font-semibold text-white/80 uppercase tracking-widest animate-pulse">Running...</span>
                  </div>
                )}
              </div>
            </section>

            {/* Submit Action */}
            <button
              onClick={() => runAnalysis()}
              disabled={loading || !image}
              className={`w-full py-3 rounded-xl font-bold uppercase tracking-wider text-xs transition-all duration-300 flex items-center justify-center gap-2 border ${
                image 
                  ? "bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 border-sky-400/20 text-white cursor-pointer hover:shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:-translate-y-[1px]" 
                  : "bg-white/5 border-white/5 text-white/20 cursor-not-allowed"
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.0" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              {loading ? "Analyzing..." : "Evaluate Profile"}
            </button>

            {/* Error Display */}
            {error && (
              <div className="p-3.5 rounded-xl border border-rose-500/20 bg-rose-500/10 text-rose-300 text-[11px] flex gap-2 items-start animate-fade-in">
                <svg className="w-4 h-4 shrink-0 mt-[1px]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.0" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <span className="leading-snug">{error}</span>
              </div>
            )}
          </div>

          {/* COLUMN 2: Decision Model Selector grids */}
          <div className="sandbox-column-middle">
            <section className="panel flex flex-col gap-4">
              <div>
                <h2 className="section-label">Decision Model Weights</h2>
                <span className="text-[10px] text-white/40 block mt-1 uppercase tracking-wider font-semibold">Select scoring environment</span>
              </div>
              
              <div className="flex flex-col gap-4">
                {/* 1. Base Epochs */}
                <div>
                  <h3 className="text-white/40 text-[10px] uppercase font-bold tracking-wider mb-2">Base Regressor Epochs</h3>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { id: "original", title: "Original (7k)", desc: "50% Ridge + 50% Multi" },
                      { id: "chandra_7k", title: "Chandra (7k Method)", desc: "50% Ridge + 50% Multi" },
                      { id: "round1", title: "Round 1", desc: "50% Ridge + 50% Multi" },
                      { id: "round2", title: "Round 2", desc: "22% Ridge + 78% Multi" },
                      { id: "round3", title: "Round 3 (Latest)", desc: "44% Ridge + 56% Multi" },
                      { id: "bumble_only", title: "Bumble R1", desc: "50% Ridge + 50% Multi" },
                      { id: "bumble_only_round2", title: "Bumble R2", desc: "22% Ridge + 78% Multi" }
                    ].map((item) => (
                      <button
                        key={item.id}
                        onClick={() => handleModelSelect(item.id)}
                        className={`text-left p-2.5 rounded-xl border transition-all duration-200 flex flex-col gap-0.5 cursor-pointer ${
                          modelVersion === item.id 
                            ? "border-sky-500/40 bg-sky-500/10 text-white shadow-[0_0_12px_rgba(56,189,248,0.15)]" 
                            : "border-white/5 bg-white/[0.01] hover:bg-white/[0.04] text-white/60 hover:text-white/90"
                        }`}
                      >
                        <div className="flex items-center justify-between w-full">
                          <span className="font-bold text-xs leading-tight">{item.title}</span>
                          {modelVersion === item.id && (
                            <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-ping shrink-0 ml-1" />
                          )}
                        </div>
                        <span className="text-[9px] opacity-60 leading-tight block mt-0.5">{item.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* 2. Experimental Veto Models */}
                <div>
                  <h3 className="text-white/40 text-[10px] uppercase font-bold tracking-wider mb-2">Trained Veto Classifiers</h3>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { id: "experimental1", title: "Experimental 1", desc: "R2 + Veto Model" },
                      { id: "experimental2", title: "Experimental 2", desc: "R3 + Veto Model" },
                      { id: "experimental3", title: "Experimental 3", desc: "Original + MultimodalX Blend" },
                      { id: "multimodalx", title: "MultimodalX", desc: "Threshold + Preference" },
                      { id: "multimodalx2", title: "MultimodalX2", desc: "Threshold + Pref (K=9)" },
                      { id: "multimodalx3", title: "MultimodalX3", desc: "R3 + MM X3 Veto" },
                      { id: "multimodalx4", title: "MultimodalX4", desc: "R3 + Blended Veto" },
                      { id: "multimodalx5", title: "MultimodalX5", desc: "R3 + Blended Veto 2" },
                      { id: "multimodalx6", title: "MultimodalX6", desc: "Round 2 + Veto Spline" }
                    ].map((item) => (
                      <button
                        key={item.id}
                        onClick={() => handleModelSelect(item.id)}
                        className={`text-left p-2.5 rounded-xl border transition-all duration-200 flex flex-col gap-0.5 cursor-pointer ${
                          modelVersion === item.id 
                            ? "border-sky-500/40 bg-sky-500/10 text-white shadow-[0_0_12px_rgba(56,189,248,0.15)]" 
                            : "border-white/5 bg-white/[0.01] hover:bg-white/[0.04] text-white/60 hover:text-white/90"
                        }`}
                      >
                        <div className="flex items-center justify-between w-full">
                          <span className="font-bold text-xs leading-tight">{item.title}</span>
                          {modelVersion === item.id && (
                            <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-ping shrink-0 ml-1" />
                          )}
                        </div>
                        <span className="text-[9px] opacity-60 leading-tight block mt-0.5">{item.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </section>
          </div>

          {/* COLUMN 3: Rich visual telemetry and results */}
          <div className="sandbox-column-right">
            
            {result ? (
              <div className="flex flex-col gap-5 animate-fade-in">
                
                {/* 1. Main Decision & Blended Card */}
                <div className="grid grid-cols-1 md:grid-cols-[1.2fr_1fr] gap-4 items-stretch">
                  
                  {/* Decision Banner */}
                  <section className="panel sandbox-overall-card flex flex-col justify-between p-6 h-full min-h-[160px] relative overflow-hidden">
                    <div className="z-10">
                      <h3 className="section-label">Engine Recommendation</h3>
                      <h2 className="overall-title uppercase text-white/50 tracking-wider">Automated Decision</h2>
                    </div>

                    <div className="flex items-end justify-between z-10 mt-4">
                      <div>
                        <span className="text-white/60 text-xs uppercase block font-semibold">Active Threshold</span>
                        <span className="font-mono text-lg font-bold text-white">{formatNumber(result.threshold)}</span>
                      </div>
                      <div>
                        <span className="text-white/60 text-xs uppercase block font-semibold text-right">Delta</span>
                        <span className={`font-mono text-sm font-bold ${result.threshold_diff >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                          {result.threshold_diff >= 0 ? "+" : ""}{formatNumber(result.threshold_diff)}
                        </span>
                      </div>
                    </div>

                    {/* Decision Glow Stamp */}
                    <div className={`absolute right-4 top-1/2 -translate-y-1/2 stamp font-mono italic select-none ${
                      result.decision === "RIGHT" 
                        ? "text-emerald-400 drop-shadow-[0_0_15px_rgba(52,211,147,0.4)]" 
                        : "text-rose-400 drop-shadow-[0_0_15px_rgba(251,113,133,0.4)]"
                    }`}>
                      {result.decision === "RIGHT" ? "SMASH" : "PASS"}
                    </div>
                  </section>

                  {/* Overall Face-Biased Blended Card */}
                  <section className="panel flex flex-col justify-between p-6 h-full min-h-[160px] bg-white/[0.02]">
                    <div>
                      <h3 className={`section-label ${(result.is_preference_mode || (result.formula_description && result.formula_description.includes("P(like)"))) ? "text-amber-300" : "text-indigo-300"}`}>
                        {result.is_preference_mode ? "Veto Layer Final Output" : (result.formula_description && result.formula_description.includes("P(like)")) ? "CLI Blended Output" : "Blended Core Metric"}
                      </h3>
                      <h2 className="overall-title">
                        {(result.is_preference_mode || (result.formula_description && result.formula_description.includes("P(like)"))) ? "CLI Final Score" : "Face-Biased Rating"}
                      </h2>
                    </div>
                    
                    <div className="flex items-baseline justify-between mt-2">
                      <span className="text-[3rem] font-black font-mono leading-none text-transparent bg-clip-text bg-gradient-to-r from-sky-400 to-indigo-400">
                        {formatNumber(result.primary_score)}
                      </span>
                      <span className="text-xs text-white/40 font-mono text-right max-w-[150px] leading-tight">
                        {result.is_preference_mode 
                          ? `P(like) ${((result.scores.preference_probability ?? 0) * 100).toFixed(1)}%, normalized to 0-100` 
                          : (result.formula_description ?? `${(result.primary_face_weight * 100).toFixed(0)}% Ridge + ${((1 - result.primary_face_weight) * 100).toFixed(0)}% Multimodal`)}
                      </span>
                    </div>
                  </section>

                </div>

                {/* 2. Grid of ALL raw score numbers */}
                <section className="panel">
                  <div className="flex justify-between items-center mb-4">
                    <h2 className="section-label text-sky-300">All Model Metrics (RAW Telemetry)</h2>
                    <span className="text-[10px] uppercase font-bold text-white/30 tracking-wider">Direct Matrix View</span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    
                    <div className="model-card border border-sky-500/20 bg-sky-500/5">
                      <h3 className="metric-kicker text-sky-300">
                        {(result.is_preference_mode || (result.formula_description && result.formula_description.includes("P(like)"))) ? "CLI Final Score" : "Blended Rating"}
                      </h3>
                      <h2 className="sandbox-metric-title text-xs text-white/85 mt-0.5 leading-tight">{result.model_display_name}</h2>
                      <span className="sandbox-metric-value font-mono text-xl text-sky-400 mt-1 block">
                        {formatNumber(result.primary_score)}
                      </span>
                      <span className="text-[9px] text-white/45 mt-1 block leading-tight">
                        {result.is_preference_mode ? "P(like) x 100 after veto layer" : (result.formula_description && result.formula_description.includes("P(like)")) ? "CLI blended core rating" : "Blended core rating"}
                      </span>
                    </div>

                    {result.is_preference_mode && (
                      <div className="model-card border border-amber-500/20 bg-amber-500/[0.04]">
                        <h3 className="metric-kicker text-amber-300">Active Base Scorer</h3>
                        <h2 className="sandbox-metric-title text-xs text-white/85 mt-0.5 leading-tight">
                          Score before veto layer
                        </h2>
                        <span className="sandbox-metric-value font-mono text-xl text-amber-300 mt-1 block">
                          {formatNumber(result.base_score)}
                        </span>
                        <span className="text-[9px] text-white/45 mt-1 block leading-tight">
                          This is the scorer value fed into the selected veto setup.
                        </span>
                      </div>
                    )}

                    {result.scores.preference_probability !== null && !result.is_preference_mode && (
                      <div className="model-card border border-amber-500/20 bg-amber-500/[0.04]">
                        <h3 className="metric-kicker text-amber-300">Preference Probability</h3>
                        <h2 className="sandbox-metric-title text-xs text-white/85 mt-0.5 leading-tight">
                          P(like) component
                        </h2>
                        <span className="sandbox-metric-value font-mono text-xl text-amber-300 mt-1 block">
                          {((result.scores.preference_probability ?? 0) * 100).toFixed(1)}%
                        </span>
                        <span className="text-[9px] text-white/45 mt-1 block leading-tight">
                          Raw preference classifier probability, used as a feature.
                        </span>
                      </div>
                    )}

                    <div className="model-card border border-indigo-500/10 bg-indigo-500/[0.02]">
                      <h3 className="metric-kicker text-indigo-300">Ridge Linear</h3>
                      <h2 className="sandbox-metric-title text-xs text-white/85 mt-0.5 leading-tight">Ridge Regressor</h2>
                      <span className="sandbox-metric-value font-mono text-xl text-indigo-400 mt-1 block">{formatNumber(result.scores.ridge)}</span>
                      <span className="text-[9px] text-white/45 mt-1 block leading-tight">Linear face-similarity score</span>
                    </div>

                    <div className="model-card border border-emerald-500/10 bg-emerald-500/[0.02]">
                      <h3 className="metric-kicker text-emerald-300">CLIP Vision</h3>
                      <h2 className="sandbox-metric-title text-xs text-white/85 mt-0.5 leading-tight">Multimodal (CLIP)</h2>
                      <span className="sandbox-metric-value font-mono text-xl text-emerald-400 mt-1 block">{formatNumber(result.scores.multimodal)}</span>
                      <span className="text-[9px] text-emerald-300/50 mt-1 block leading-tight">{result.comparison_text.multimodal}</span>
                    </div>

                    <div className="model-card border border-rose-500/10 bg-rose-500/[0.02]">
                      <h3 className="metric-kicker text-rose-300">KNN Similarity</h3>
                      <h2 className="sandbox-metric-title text-xs text-white/85 mt-0.5 leading-tight">K={result.primary_k} Matches</h2>
                      <span className="sandbox-metric-value font-mono text-xl text-rose-400 mt-1 block">{formatNumber(result.scores.knn)}</span>
                      <span className="text-[9px] text-rose-300/50 mt-1 block leading-tight">{result.comparison_text.knn}</span>
                    </div>

                  </div>

                  <div className="flex items-center justify-between gap-3 mt-4 mb-2">
                    <h3 className="text-white/45 text-[10px] uppercase font-bold tracking-wider">Base Epoch Comparison</h3>
                    <span className="text-[9px] text-white/35 text-right">Comparison-only scores unless that setup is selected.</span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                    {[
                      { key: "original_face_biased", name: "Original Model", kicker: "Legacy (7K)" },
                      { key: "chandra_7k_face_biased", name: "Chandra Model", kicker: "New Labels (7K Method)" },
                      { key: "round1_face_biased", name: "Round 1 Model", kicker: "Epoch R1" },
                      { key: "round2_face_biased", name: "Round 2 Model", kicker: "Epoch R2" },
                      { key: "round3_face_biased", name: "Round 3 Model", kicker: "Epoch R3 (Latest)" },
                      { key: "bumble_only_face_biased", name: "Bumble Only R1", kicker: "Epoch B1" },
                      { key: "bumble_only_round2_face_biased", name: "Bumble Only R2", kicker: "Epoch B2" }
                    ].map((cfg) => {
                      const score = (result.scores as Record<string, number | null>)[cfg.key];
                      const compText = result.comparison_text[cfg.key];
                      const isSelectedBaseCard = selectedBaseCardKey[result.model_version] === cfg.key;
                      return (
                        <div
                          key={cfg.key}
                          className={`model-card border ${
                            isSelectedBaseCard
                              ? "border-amber-400/30 bg-amber-400/[0.05]"
                              : "border-white/5 bg-white/[0.01]"
                          }`}
                        >
                          <div className="flex items-center justify-between gap-1">
                            <h3 className="metric-kicker text-[8px] text-white/40">{cfg.kicker}</h3>
                            {isSelectedBaseCard && (
                              <span className="text-[7px] uppercase font-bold tracking-wider text-amber-300">Selected base</span>
                            )}
                          </div>
                          <h2 className="sandbox-metric-title text-[10px] text-white/70 block mt-0.5 leading-tight">{cfg.name}</h2>
                          <div className="flex flex-col gap-0.5 mt-1.5">
                            <span className="sandbox-metric-value font-mono text-base text-white/95 leading-none">{formatNumber(score)}</span>
                            <span className={`text-[8px] font-mono leading-none ${
                              (score ?? 0) >= (result.scores.ridge ?? 0) ? "text-emerald-400" : "text-rose-400"
                            }`}>{compText}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </section>

                {/* 3. Nearest Neighbor References Details */}
                <div className="grid grid-cols-1 md:grid-cols-[1fr_1.8fr] gap-5 items-start">
                  
                  {/* Similarity Metrics */}
                  <section className="panel flex flex-col gap-4">
                    <h2 className="section-label">Vector Similarities</h2>
                    <div className="flex flex-col gap-4">
                      
                      <div className="p-3 bg-white/[0.02] rounded-xl border border-white/5">
                        <span className="text-white/40 text-[10px] uppercase font-bold tracking-wider block">Max Reference Similarity</span>
                        <span className="text-xl font-bold font-mono text-sky-300 block mt-1">{result.details.max_similarity.toFixed(4)}</span>
                        <span className="text-[9px] text-white/30 block mt-1">Strongest matching reference vector</span>
                      </div>

                      <div className="p-3 bg-white/[0.02] rounded-xl border border-white/5">
                        <span className="text-white/40 text-[10px] uppercase font-bold tracking-wider block">Mean Reference Similarity</span>
                        <span className="text-xl font-bold font-mono text-indigo-300 block mt-1">{result.details.mean_similarity.toFixed(4)}</span>
                        <span className="text-[9px] text-white/30 block mt-1">Average of top 20 matches</span>
                      </div>

                      <div className="p-3 bg-white/[0.02] rounded-xl border border-white/5">
                        <span className="text-white/40 text-[10px] uppercase font-bold tracking-wider block">Reference Corpus Size</span>
                        <span className="text-xl font-bold font-mono text-white block mt-1">{result.details.references_count.toLocaleString()}</span>
                        <span className="text-[9px] text-white/30 block mt-1">Total indexed profile vectors</span>
                      </div>

                    </div>
                  </section>

                  {/* Nearest Reference Profiles List */}
                  <section className="panel flex flex-col gap-4">
                    <h2 className="section-label">Nearest Matching References</h2>
                    <div className="max-h-[300px] overflow-y-auto pr-1 flex flex-col gap-2 custom-scrollbar">
                      {result.nearest_references.map((ref, idx) => (
                        <div 
                          key={idx} 
                          className="flex items-center justify-between p-3 rounded-lg bg-white/[0.01] hover:bg-white/[0.03] border border-white/5 transition-all text-xs font-mono"
                        >
                          <div className="flex gap-2 items-center min-w-0">
                            <span className="text-white/30 w-4 font-bold">{(idx + 1).toString().padStart(2, "0")}</span>
                            <span className="text-white/80 truncate text-xs" title={ref.name}>{ref.name}</span>
                          </div>
                          
                          <div className="flex items-center gap-4 shrink-0">
                            <div className="text-right">
                              <span className="text-white/30 text-[9px] uppercase block tracking-wider font-bold">Rating</span>
                              <span className="text-sky-300 font-bold">{ref.rating.toFixed(0)}</span>
                            </div>
                            <div className="text-right w-16">
                              <span className="text-white/30 text-[9px] uppercase block tracking-wider font-bold">Similarity</span>
                              <span className="text-emerald-400 font-bold">{ref.similarity.toFixed(4)}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>

                </div>

              </div>
            ) : (
              <div className="w-full h-full border border-dashed border-white/10 rounded-2xl flex flex-col items-center justify-center p-12 text-center text-white/30 min-h-[450px]">
                <svg className="w-16 h-16 mb-4 text-white/15" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2z" />
                </svg>
                <h3 className="font-semibold text-white/50 text-sm">Telemetry Display Idle</h3>
                <p className="text-xs text-white/35 max-w-sm mt-1">Upload a profile picture and click Evaluate Profile. The full vector metrics and regressor ratings will populate here instantly.</p>
              </div>
            )}

          </div>

        </div>

      </div>
    </main>
  );
}
