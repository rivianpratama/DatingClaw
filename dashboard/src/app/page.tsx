"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

type ScoreData = {
  averages: { score: number; face_biased: number; multimodal: number; ridge: number; knn: number; };
  actionAverages: { left: number; right: number; };
  swipes: { left: number; right: number; leftPercent: number; rightPercent: number; };
  methodDistribution: Record<string, number>;
  scoreDistribution: Record<string, number>;
  velocity: number;
  latest: { final_score: number | null; base_score: number | null; face_biased: number | null; multimodal: number | null; ridge: number | null; knn: number | null; action: string | null; setup_name: string | null; method: string | null; decision_mode: string | null; preference_probability: number | null; preference_threshold: number | null; divergence: number | null; screenshot: string | null; } | null;
  activeSetup?: { name: string | null; decisionMode: string; preferenceModelPath: string | null; faceWeight: string | null };
  dynamicThreshold: {
    enabled: boolean;
    active: boolean;
    threshold: number;
    rawThreshold: number | null;
    fallbackThreshold: number;
    targetRightRate: number;
    window: number;
    minHistory: number;
    minThreshold: number;
    maxThreshold: number;
    historyCount: number;
    projectedRightCount: number;
    projectedRightPercent: number;
    actualLeftCount: number;
    actualRightCount: number;
    actualRightPercent: number;
  };
  dynamicPreferenceThreshold?: {
    enabled: boolean;
    active: boolean;
    threshold: number;
    rawThreshold: number | null;
    fallbackThreshold: number;
    targetRightRate: number;
    window: number;
    minHistory: number;
    minThreshold: number;
    maxThreshold: number;
    historyCount: number;
    projectedRightCount: number;
    projectedRightPercent: number;
    actualLeftCount: number;
    actualRightCount: number;
    actualRightPercent: number;
  };
  trend?: { time: number; threshold: number; score: number | null; action: string | null; globalAverage?: number | null }[];
  records: number;
  v6?: {
    model: string;
    scoring: string;
    threshold: number;
    latest: { gambler_score: number | null; v6r5_z: number | null; lora_z: number | null; psi_sync: number | null; threshold: number | null; action: string | null; screenshot: string | null; inference_ms: number | null; capture_ms: number | null; score_ms: number | null; swipe_ms: number | null; cycle_ms: number | null; } | null;
    averages: Record<string, number | null>;
    latency: Record<string, { p50: number | null; p95: number | null }>;
    counts: { left: number; right: number; total: number };
    right_rate: number;
    velocity: number;
    records: number;
  } | null;
};

export default function Dashboard() {
  const [data, setData] = useState<ScoreData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sessionOffset, setSessionOffset] = useState<number>(0);
  const [viewMode, setViewMode] = useState<'all' | 'session'>('all');
  const [totalRecords, setTotalRecords] = useState<number>(0);
  const [trendWindow, setTrendWindow] = useState<number>(30 * 60 * 1000);
  const [currentTime, setCurrentTime] = useState<number>(1779344400000);
  const [mounted, setMounted] = useState(false);
  const [showThreshold, setShowThreshold] = useState(true);
  const [showAverage, setShowAverage] = useState(true);

  const handleSetViewMode = (mode: 'all' | 'session') => {
    setViewMode(mode);
    localStorage.setItem('bumble_view_mode', mode);
  };

  const handleReset = () => {
    setSessionOffset(totalRecords);
    setViewMode('session');
    localStorage.setItem('bumble_session_offset', totalRecords.toString());
    localStorage.setItem('bumble_view_mode', 'session');
  };

  // Safe client-side loading after mounting to prevent React hydration mismatch
  useEffect(() => {
    const savedOffset = localStorage.getItem('bumble_session_offset');
    if (savedOffset) {
      setSessionOffset(parseInt(savedOffset, 10));
    }
    const savedMode = localStorage.getItem('bumble_view_mode');
    if (savedMode === 'session' || savedMode === 'all') {
      setViewMode(savedMode);
    }
    setCurrentTime(Date.now());
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    const fetchData = async () => {
      try {
        const currentOffset = viewMode === 'session' ? sessionOffset : 0;
        const res = await fetch(`/api/scores?offset=${currentOffset}`);
        if (!res.ok) throw new Error("Failed to fetch data");
        const json = await res.json();
        if (json.error) setError(json.error);
        else { 
          setData(json); 
          setTotalRecords(json.totalRecords);
          setError(null); 
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, [viewMode, sessionOffset, mounted]);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(Date.now());
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const formatNumber = (num: number | null | undefined) => num === null || num === undefined ? "--" : num.toFixed(2);

  if (error) return <div className="error-state">{error}</div>;
  if (!data) return <div className="loading-state"><div className="loading-spinner"></div></div>;

  const isPreferenceMode = data.activeSetup?.decisionMode === 'preference';
  const v6 = data.v6;
  const activeThreshold = isPreferenceMode && data.dynamicPreferenceThreshold ? data.dynamicPreferenceThreshold : data.dynamicThreshold;
  const thresholdLabel = v6?.model === 'v6.8' ? 'Fixed Threshold' : 'Final Threshold';
  const finalThresholdScale = isPreferenceMode ? 100 : 1;
  const topMethods = Object.entries(data.methodDistribution).sort((a, b) => b[1] - a[1]);
  const modelMetrics = v6 ? [
    { key: 'face_biased', title: 'v6r5 Z', subtitle: 'Tight + Wide', color: '#a78bfa' },
    { key: 'multimodal', title: 'LoRA Z', subtitle: 'Photo LoRA', color: '#38bdf8' },
    { key: 'ridge', title: 'Synergy', subtitle: 'Agreement', color: '#fbbf24' },
    { key: 'knn', title: 'Gambler', subtitle: '30/70 Blend', color: '#fb7185' },
  ] : [
    { key: 'face_biased', title: 'Base Face Biased', subtitle: 'Pre-Veto Blend', color: '#a78bfa' },
    { key: 'multimodal', title: 'Multimodal', subtitle: 'Vision + Text', color: '#38bdf8' },
    { key: 'ridge', title: 'Ridge', subtitle: 'Linear Model', color: '#fbbf24' },
    { key: 'knn', title: 'k-NN', subtitle: 'Similarity', color: '#fb7185' },
  ];
  const trendOptions = [
    { l: '1m', v: 60 * 1000 },
    { l: '5m', v: 5 * 60 * 1000 },
    { l: '30m', v: 30 * 60 * 1000 },
    { l: '1h', v: 60 * 60 * 1000 },
    { l: '6h', v: 6 * 60 * 60 * 1000 },
    { l: '12h', v: 12 * 60 * 60 * 1000 },
    { l: '1d', v: 24 * 60 * 60 * 1000 },
  ];

  return (
    <main className="dashboard-shell selection:bg-sky-500/30">
      <div className="dashboard-frame">
        <header className="dashboard-header">
          <div>
            <div className="dashboard-status">
              <span className="dashboard-status-dot" />
              Operational Telemetry
            </div>
            <h1 className="dashboard-title">BumbleLog Matrix</h1>
          </div>
          <div className="dashboard-actions">
            <div className="summary-card compact">
              <span className="summary-label">{thresholdLabel}</span>
              <span className="summary-value" style={{ color: 'var(--amber)' }}>{formatNumber(activeThreshold.threshold * finalThresholdScale)}</span>
            </div>
            <div className="summary-card compact">
              <span className="summary-label">Setup</span>
              <span className="summary-value" style={{ color: 'var(--indigo)', textTransform: 'uppercase' }}>{data.activeSetup?.name || topMethods[0]?.[0] || 'N/A'}</span>
            </div>
            <div className="summary-card compact">
              <span className="summary-label">Divergence</span>
              <span className="summary-value" style={{ color: '#c084fc' }}>{formatNumber(data.latest?.divergence)}</span>
            </div>
            <div className="summary-card">
              <span className="summary-label">Processing Speed</span>
              <div>
                <span className="summary-value">{data.velocity.toFixed(1)}</span>
                <span className="text-white/55 font-mono text-xs"> swipes/min</span>
              </div>
            </div>
            <div className="summary-card">
              <span className="summary-label">Total Processed</span>
              <span className="summary-value">{totalRecords.toLocaleString()}</span>
            </div>
            <Link href="/sandbox" className="gallery-link">
              <span className="summary-label">Manual Test</span>
              <span className="gallery-title">Profile Sandbox</span>
            </Link>
            <Link href="/gallery" className="gallery-link">
              <span className="summary-label">Access Repository</span>
              <span className="gallery-title">View Gallery</span>
            </Link>
          </div>
        </header>

        <div className="dashboard-grid">
          <section className="preview-card">
            <h2 className="section-label">Live Signal Intercept</h2>
              {data.latest?.screenshot ? (
              <div className="preview-media">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img 
                    src={`/api/image?file=${encodeURIComponent(data.latest.screenshot)}`} 
                    alt="Latest Profile"
                  className="preview-image"
                  />
                <div className="preview-shade" />
                  
                <div className="preview-metrics">
                    {(v6 ? [{l:'v6r5', v:data.latest.face_biased}, {l:'LoRA', v:data.latest.multimodal}, {l:'Agree', v:data.latest.ridge}, {l:'Gambler', v:data.latest.knn}] : [{l:'Base', v:data.latest.base_score}, {l:'Multi', v:data.latest.multimodal}, {l:'Ridge', v:data.latest.ridge}, {l:'KNN', v:data.latest.knn}]).map(m => (
                    <div key={m.l} className="mini-metric">
                      <span className="mini-metric-label">{m.l}</span>
                      <span className="mini-metric-value">{formatNumber(m.v)}</span>
                       </div>
                    ))}
                  </div>

                <div className="preview-footer">
                  <div>
                    <span className="score-caption">{v6 ? 'Gambler Score' : 'Final Score'}</span>
                    <span className="score-value">
                          {formatNumber(data.latest.final_score)}
                        </span>
                      </div>
                      {data.latest.action && (
                    <div key={data.latest.screenshot || 'empty'} className={`stamp animate-pop-stamp ${data.latest.action === 'right' ? 'text-emerald-400' : 'text-rose-500'}`}>
                          {data.latest.action === 'right' ? 'OK' : 'PASS'}
                        </div>
                      )}
                    </div>
                </div>
              ) : (
              <div className="empty-preview">
                <span className="section-label animate-pulse">Awaiting Signal...</span>
                </div>
              )}
          </section>

          <div className="telemetry-column">
            <div className="top-panels">
              <section className="panel">
                <div className="panel-header">
                  <h2 className="section-label">Decision Engine & Thresholds</h2>
                  <div className="control-group">
                    <button 
                      onClick={() => handleSetViewMode('all')}
                      className={`control-button ${viewMode === 'all' ? 'active' : ''}`}
                    >
                      All-Time
                    </button>
                    <button 
                      onClick={() => handleSetViewMode('session')}
                      className={`control-button ${viewMode === 'session' ? 'active' : ''}`}
                    >
                      Latest Setup
                    </button>
                    <button 
                      onClick={handleReset}
                      className="control-button danger"
                      title="Reset Stats"
                    >
                      Reset
                    </button>
                  </div>
                </div>

                <div className="decision-split">
                  <div className="decision-block">
                    <span className="decision-percent text-rose-400">{data.swipes.leftPercent.toFixed(1)}%</span>
                    <span className="decision-meta">Left Swipes ({data.swipes.left})</span>
                    <span className="decision-chip text-rose-300 bg-rose-500/10 border border-rose-500/20">Avg Score: {formatNumber(data.actionAverages.left)}</span>
                  </div>
                  <div className="decision-block right">
                    <span className="decision-percent text-emerald-400">{data.swipes.rightPercent.toFixed(1)}%</span>
                    <span className="decision-meta">Right Swipes ({data.swipes.right})</span>
                    <span className="decision-chip text-emerald-300 bg-emerald-500/10 border border-emerald-500/20">Avg Score: {formatNumber(data.actionAverages.right)}</span>
                  </div>
                </div>
                <div className="swipe-bar">
                  <div className="swipe-segment bg-gradient-to-r from-rose-600 to-rose-400" style={{ width: `${data.swipes.leftPercent}%` }} />
                  <div className="swipe-segment bg-gradient-to-r from-emerald-400 to-emerald-600" style={{ width: `${data.swipes.rightPercent}%` }} />
                </div>
              </section>

              <section className="panel">
                <h2 className="section-label text-blue-300">Final Score Distribution</h2>
                <div className="distribution-bars">
                     {Object.entries(data.scoreDistribution || {}).map(([label, count]) => {
                        const maxCount = Math.max(...Object.values(data.scoreDistribution || {}), 1);
                        const heightPct = (count / maxCount) * 100;
                        return (
                      <div key={label} className="distribution-column">
                        <span className="bar-count">{count}</span>
                        <div className="bar-shell">
                          <div className="bar-fill" style={{ height: `${heightPct}%` }} />
                             </div>
                        <span className="bar-label">{label}</span>
                          </div>
                        )
                     })}
                  </div>
              </section>
            </div>

            {data.trend && data.trend.length > 0 && (
              <section className="panel trend-panel">
                <div className="panel-header animate-fade-in">
                  <div className="trend-title">
                    <h2 className="section-label text-amber-300">{isPreferenceMode ? 'Dynamic Preference Trend' : 'Dynamic Threshold Trend'}</h2>
                    <p className="trend-subtitle">{isPreferenceMode ? 'Real-time P(like) percentile over the active setup' : 'Real-time sliding window calculations over the active timeframe'}</p>
                  </div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <div className="control-group">
                      <button 
                        onClick={() => setShowThreshold(!showThreshold)}
                        className={`control-button ${showThreshold ? 'active text-emerald-400 border-emerald-500/30 bg-emerald-500/15' : ''}`}
                        title="Toggle Threshold Trend"
                      >
                        Threshold
                      </button>
                      <button 
                        onClick={() => setShowAverage(!showAverage)}
                        className={`control-button ${showAverage ? 'active text-indigo-400 border-indigo-500/30 bg-indigo-500/15' : ''}`}
                        title="Toggle Global Average Trend"
                      >
                        Global Average
                      </button>
                    </div>

                    <div className="control-group">
                      {trendOptions.map(opt => (
                        <button 
                          key={opt.l} 
                          onClick={() => setTrendWindow(opt.v)}
                          className={`control-button ${trendWindow === opt.v ? 'active text-amber-300 border-amber-500/30 bg-amber-500/15' : ''}`}
                        >
                          {opt.l}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
                
                <div className="trend-chart">
                  {(() => {
                    const now = currentTime || (data.trend && data.trend.length > 0 ? data.trend[data.trend.length - 1].time : 1779344400000);
                    const minTime = now - trendWindow;
                    const filteredTrend = data.trend ? data.trend.filter(t => t.time >= minTime) : [];
                    
                    const trendScale = isPreferenceMode ? 100 : 1;
                    const currentVal = (data.trend.length > 0 ? data.trend[data.trend.length - 1].threshold : 55) * trendScale;
                    const currentAvgVal = (data.trend.length > 0 ? (data.trend[data.trend.length - 1].globalAverage ?? data.trend[data.trend.length - 1].threshold) : 55) * trendScale;

                    // Downsample the data into 80 time-based buckets (approx every 10px on a standard screen)
                    const bucketCount = 80;
                    const bucketWidth = trendWindow / bucketCount;
                    const sampledTrend: typeof filteredTrend = [];
                    
                    for (let i = 0; i < bucketCount; i++) {
                      const bucketStart = minTime + i * bucketWidth;
                      const bucketEnd = bucketStart + bucketWidth;
                      
                      const pointsInBucket = filteredTrend.filter(t => t.time >= bucketStart && t.time < bucketEnd);
                      if (pointsInBucket.length > 0) {
                        sampledTrend.push(pointsInBucket[pointsInBucket.length - 1]);
                      } else {
                        const lastKnownPoint = filteredTrend.filter(t => t.time < bucketEnd).pop();
                        if (lastKnownPoint) {
                          sampledTrend.push({
                            time: bucketEnd,
                            threshold: lastKnownPoint.threshold,
                            score: null,
                            action: null,
                            globalAverage: lastKnownPoint.globalAverage
                          });
                        }
                      }
                    }

                    // Fallback to a flat line if there's no data in the active timescale window
                    const displayTrend = sampledTrend.length >= 2
                      ? sampledTrend
                      : [
                          { time: minTime, threshold: currentVal / trendScale, score: null, action: null, globalAverage: currentAvgVal / trendScale },
                          { time: now, threshold: currentVal / trendScale, score: null, action: null, globalAverage: currentAvgVal / trendScale }
                        ];

                    const thresholdValues = showThreshold ? displayTrend.map(t => t.threshold * trendScale) : [];
                    const averageValues = showAverage ? displayTrend.map(t => (t.globalAverage ?? t.threshold) * trendScale) : [];
                    const allVisibleValues = [...thresholdValues, ...averageValues];

                    const maxThresholdVal = allVisibleValues.length > 0 ? Math.max(...allVisibleValues) : (isPreferenceMode ? 100 : 70);
                    const minThresholdVal = allVisibleValues.length > 0 ? Math.min(...allVisibleValues) : (isPreferenceMode ? 0 : 45);

                    // Zoom in tightly around the min and max threshold values of the timescale
                    const chartPadding = 2;
                    const chartMax = 100;
                    let minThreshold = Math.max(0, minThresholdVal - chartPadding);
                    let maxThreshold = Math.min(chartMax, maxThresholdVal + chartPadding);
                    if (maxThreshold === minThreshold) {
                      const fallbackPadding = 5;
                      minThreshold = Math.max(0, minThreshold - fallbackPadding);
                      maxThreshold = Math.min(chartMax, maxThreshold + fallbackPadding);
                    }

                    const points = displayTrend.map((t) => {
                      const x = Math.min(100, ((t.time - minTime) / trendWindow) * 100);
                      const y = 100 - (((t.threshold * trendScale) - minThreshold) / (maxThreshold - minThreshold)) * 100;
                      return `${x},${y}`;
                    }).join(' ');

                    const avgPoints = displayTrend.map((t) => {
                      const x = Math.min(100, ((t.time - minTime) / trendWindow) * 100);
                      const y = 100 - ((((t.globalAverage ?? t.threshold) * trendScale) - minThreshold) / (maxThreshold - minThreshold)) * 100;
                      return `${x},${y}`;
                    }).join(' ');

                    const scorePoints = displayTrend.filter(t => t.score !== null).map((t) => {
                      const x = Math.min(100, ((t.time - minTime) / trendWindow) * 100);
                      const y = 100 - (((t.score! * trendScale) - minThreshold) / (maxThreshold - minThreshold)) * 100;
                      return { x, y, action: t.action };
                    });

                    const firstX = Math.min(100, ((displayTrend[0].time - minTime) / trendWindow) * 100);
                    const lastX = Math.min(100, ((displayTrend[displayTrend.length - 1].time - minTime) / trendWindow) * 100);
                    const lastY = 100 - (((displayTrend[displayTrend.length - 1].threshold * trendScale) - minThreshold) / (maxThreshold - minThreshold)) * 100;
                    const lastAvgY = 100 - (((displayTrend[displayTrend.length - 1].globalAverage ?? displayTrend[displayTrend.length - 1].threshold) * trendScale - minThreshold) / (maxThreshold - minThreshold)) * 100;

                    const areaPath = `M ${firstX},100 L ${points} L ${lastX},100 Z`;
                    const avgAreaPath = `M ${firstX},100 L ${avgPoints} L ${lastX},100 Z`;

                    // Stock market trend math
                    const startVal = displayTrend[0].threshold * trendScale;
                    const isUpTrend = currentVal >= startVal;
                    const strokeColor = isUpTrend ? '#10b981' : '#f43f5e';
                    const gradientColor = isUpTrend ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)';
                    const glowClass = isUpTrend ? 'drop-shadow-[0_0_6px_rgba(16,185,129,0.5)]' : 'drop-shadow-[0_0_6px_rgba(244,63,94,0.5)]';
                    const nowColorClass = isUpTrend ? 'text-emerald-400' : 'text-rose-400';
                    const nowLabelColorClass = isUpTrend ? 'text-emerald-500' : 'text-rose-500';

                    // Global Average visual styles
                    const avgStrokeColor = '#818cf8';
                    const avgGradientColor = 'rgba(129, 140, 248, 0.15)';
                    const avgGlowClass = 'drop-shadow-[0_0_6px_rgba(129,140,248,0.5)]';

                    return (
                      <>
                        <div className="trend-axis select-none">
                          <div>
                            <span className="chart-axis-label">Max</span>
                            <span className="axis-value">{maxThresholdVal.toFixed(1)}</span>
                          </div>
                          {showThreshold && (
                            <div>
                              <span className={`chart-axis-label ${nowLabelColorClass}`}>Now (Thresh)</span>
                              <span className={`axis-value now ${nowColorClass}`}>{currentVal.toFixed(1)}</span>
                            </div>
                          )}
                          {showAverage && (
                            <div>
                              <span className="chart-axis-label text-indigo-400">Now (Avg)</span>
                              <span className="axis-value now text-indigo-400">{currentAvgVal.toFixed(1)}</span>
                            </div>
                          )}
                          <div>
                            <span className="chart-axis-label">Min</span>
                            <span className="axis-value">{minThresholdVal.toFixed(1)}</span>
                          </div>
                        </div>

                        <div className="trend-svg-wrap">
                          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="trend-svg py-2">
                            <defs>
                              <linearGradient id="threshold-area-gradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor={gradientColor} />
                                <stop offset="100%" stopColor="rgba(255, 255, 255, 0.0)" />
                              </linearGradient>
                              <linearGradient id="average-area-gradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor={avgGradientColor} />
                                <stop offset="100%" stopColor="rgba(255, 255, 255, 0.0)" />
                              </linearGradient>
                            </defs>

                            <line x1="0" y1="25" x2="100" y2="25" stroke="rgba(255,255,255,0.02)" strokeWidth="0.2" strokeDasharray="1,1" />
                            <line x1="0" y1="50" x2="100" y2="50" stroke="rgba(255,255,255,0.05)" strokeWidth="0.2" />
                            <line x1="0" y1="75" x2="100" y2="75" stroke="rgba(255,255,255,0.02)" strokeWidth="0.2" strokeDasharray="1,1" />
                            
                            {showThreshold && <path d={areaPath} fill="url(#threshold-area-gradient)" className="transition-all duration-300" />}
                            {showAverage && <path d={avgAreaPath} fill="url(#average-area-gradient)" className="transition-all duration-300" />}
                            
                            {showThreshold && <polyline points={points} fill="none" stroke={strokeColor} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" className={`${glowClass} transition-all duration-300`} />}
                            {showAverage && <polyline points={avgPoints} fill="none" stroke={avgStrokeColor} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" className={`${avgGlowClass} transition-all duration-300`} />}

                            {scorePoints.map((pt, i) => (
                               <line 
                                 key={i} 
                                 x1={pt.x} 
                                 y1="97" 
                                 x2={pt.x} 
                                 y2="100" 
                                 stroke={pt.action === 'right' ? '#10b981' : pt.action === 'left' ? '#f43f5e' : '#4b5563'} 
                                 strokeWidth="0.5"
                                 opacity="0.25"
                                 vectorEffect="non-scaling-stroke"
                               />
                            ))}
                          </svg>

                          <div className="absolute inset-0 py-2 pointer-events-none z-20">
                            <div className="chart-label top-[25%] select-none">
                              {(maxThreshold - (maxThreshold - minThreshold) * 0.25).toFixed(1)}
                            </div>
                            <div className="chart-label top-[50%] select-none">
                              {(maxThreshold - (maxThreshold - minThreshold) * 0.5).toFixed(1)}
                            </div>
                            <div className="chart-label top-[75%] select-none">
                              {(maxThreshold - (maxThreshold - minThreshold) * 0.75).toFixed(1)}
                            </div>

                            {showThreshold && (
                              <div 
                                className="live-dot z-30 flex items-center justify-center transition-all duration-300"
                                style={{ 
                                  left: `${lastX}%`, 
                                  top: `${lastY}%`
                                }}
                              >
                                <span 
                                  className="absolute w-full h-full rounded-full opacity-100"
                                  style={{ backgroundColor: strokeColor, boxShadow: `0 0 10px ${strokeColor}` }}
                                />
                                <span 
                                  className="absolute w-full h-full rounded-full animate-ping opacity-75"
                                  style={{ border: `2px solid ${strokeColor}` }}
                                />
                              </div>
                            )}

                            {showAverage && (
                              <div 
                                className="live-dot z-30 flex items-center justify-center transition-all duration-300"
                                style={{ 
                                  left: `${lastX}%`, 
                                  top: `${lastAvgY}%`
                                }}
                              >
                                <span 
                                  className="absolute w-full h-full rounded-full opacity-100 bg-indigo-500"
                                  style={{ boxShadow: `0 0 10px #818cf8` }}
                                />
                                <span 
                                  className="absolute w-full h-full rounded-full animate-ping opacity-75 border-2 border-indigo-500"
                                />
                              </div>
                            )}
                          </div>
                        </div>
                      </>
                    );
                  })()}
                </div>
              </section>
            )}

            <section className="model-section">
              <div className="panel overall-card">
                <div>
                  <h3 className="section-label">{v6 ? `${v6.model} Gambler` : 'CLI Final Score'}</h3>
                  <h2 className="overall-title">Final Output Average</h2>
                </div>
                <span className="overall-value font-mono">
                  {formatNumber(data.averages.score)}
                </span>
              </div>

              <div className="model-grid">
                {modelMetrics.map((metric) => {
                  const avg = data.averages[metric.key as keyof typeof data.averages];
                  return (
                    <div key={metric.key} className="model-card">
                      <div>
                        <h3 className="metric-kicker">{metric.subtitle}</h3>
                        <h2 className="metric-title">{metric.title}</h2>
                      </div>
                      <span className="metric-value" style={{ color: metric.color }}>
                          {formatNumber(avg)}
                        </span>
                    </div>
                  );
                })}
              </div>
            </section>
          </div>
        </div>
      </div>
    </main>
  );
}
