"use client";
import { useEffect, useState, useRef, useCallback } from "react";
import Link from "next/link";
import DissectOverlay, { BoxData } from "@/components/DissectOverlay";

type Profile = {
  id: number;
  timestamp: string;
  screenshot: string;
  score: number;
  action: string;
  face_biased: number | null;
  multimodal: number | null;
  ridge: number | null;
  knn: number | null;
  setup_name: string;
  decision_mode?: string;
  preference_probability?: number | null;

  // v6 fields
  scoring?: string;
  gambler_score?: number | null;
  v6r5_z?: number | null;
  lora_z?: number | null;
  psi_sync?: number | null;
  cycle_ms?: number | null;
  inference_ms?: number | null;

  // Extended telemetry fields
  method?: string;
  store_path?: string;
  regressor_path?: string;
  multimodal_regressor_path?: string;
  threshold?: number | null;
  preference_model_path?: string;
  preference_threshold?: number | null;
  provider?: string;
  delay?: number | null;
  k?: number | null;
  face_weight?: number | null;
  mode_247?: boolean;

  // Dynamic standard threshold columns
  dynamic_enabled?: boolean;
  dynamic_mode?: string;
  dynamic_window?: number | null;
  dynamic_target_right_rate?: number | null;
  dynamic_percentile?: number | null;
  dynamic_min_history?: number | null;
  dynamic_min_threshold?: number | null;
  dynamic_max_threshold?: number | null;

  // Dynamic preference threshold columns
  dynamic_preference_enabled?: boolean;
  dynamic_preference_mode?: string;
  dynamic_preference_window?: number | null;
  dynamic_preference_target_right_rate?: number | null;
  dynamic_preference_percentile?: number | null;
  dynamic_preference_min_history?: number | null;
  dynamic_preference_min_threshold?: number | null;
  dynamic_preference_max_threshold?: number | null;
};

export default function GalleryPage() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const limit = 50;
  const [hasMore, setHasMore] = useState(true);
  const [activeProfile, setActiveProfile] = useState<Profile | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [totalCount, setTotalCount] = useState<number>(0);
  
  // Safe post-mount states to avoid hydration mismatch
  const [sortBy, setSortBy] = useState<'latest' | 'oldest' | 'attractive_desc' | 'attractive_asc'>('latest');
  const [isRealTime, setIsRealTime] = useState<boolean>(false);
  const [setupFilter, setSetupFilter] = useState<string>('');
  const [actionFilter, setActionFilter] = useState<string>('');
  const [availableSetups, setAvailableSetups] = useState<string[]>([]);
  const [mounted, setMounted] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [dissectMode, setDissectMode] = useState<boolean>(true);
  const [showFace, setShowFace] = useState<boolean>(true);
  const [showContext, setShowContext] = useState<boolean>(true);
  const [showLandmarks, setShowLandmarks] = useState<boolean>(true);
  const [boxesMap, setBoxesMap] = useState<Record<string, BoxData>>({});

  const observerTarget = useRef<HTMLDivElement | null>(null);
  const fetchingRef = useRef(false);
  const pageRef = useRef(1);

  const fetchProfiles = useCallback(async (
    pageToFetch: number, 
    append = false, 
    currentSortBy = sortBy, 
    silent = false, 
    currentSetupFilter = setupFilter,
    currentActionFilter = actionFilter
  ) => {
    if (fetchingRef.current && append) return;
    try {
      if (!silent) {
        setLoading(true);
      }
      fetchingRef.current = true;
      const offset = (pageToFetch - 1) * limit;
      const res = await fetch(`/api/history?limit=${limit}&offset=${offset}&sortBy=${currentSortBy}&setupName=${encodeURIComponent(currentSetupFilter)}&action=${currentActionFilter}`);
      if (!res.ok) return;
      const json = await res.json();
      if (json.data) {
        if (append) {
           setProfiles(prev => {
             const existingIds = new Set(prev.map(p => p.id));
             const newItems = json.data.filter((p: Profile) => !existingIds.has(p.id));
             return [...prev, ...newItems];
           });
        } else {
           if (silent) {
             if (currentSortBy === 'latest') {
               setProfiles(prev => {
                 const existingIds = new Set(prev.map(p => p.id));
                 const newItems = json.data.filter((p: Profile) => !existingIds.has(p.id));
                 if (newItems.length === 0) return prev;
                 return [...newItems, ...prev];
               });
             } else {
               if (pageToFetch === 1) {
                 setProfiles(json.data);
               }
             }
           } else {
             setProfiles(json.data);
           }
         }
         setHasMore(json.data.length === limit);
         setTotalCount(json.total || 0);
      }
      if (json.uniqueSetups) {
        setAvailableSetups(json.uniqueSetups);
      }
    } catch (e) {
      console.error(e);
    } finally {
      if (!silent) {
        setLoading(false);
      }
      fetchingRef.current = false;
    }
  }, [limit, sortBy, setupFilter, actionFilter]);

  // Load localStorage state safely after mount
  useEffect(() => {
    const savedSort = localStorage.getItem('bumble_gallery_sort');
    if (savedSort) setSortBy(savedSort as any);
    
    const savedRealtime = localStorage.getItem('bumble_gallery_realtime');
    if (savedRealtime) setIsRealTime(savedRealtime === 'true');
    
    const savedSetup = localStorage.getItem('bumble_gallery_setup_filter');
    if (savedSetup) setSetupFilter(savedSetup);

    const savedAction = localStorage.getItem('bumble_gallery_action_filter');
    if (savedAction) setActionFilter(savedAction);

    const savedDissect = localStorage.getItem('bumble_gallery_dissect');
    if (savedDissect !== null) setDissectMode(savedDissect === 'true');
    const savedFace = localStorage.getItem('bumble_gallery_show_face');
    if (savedFace !== null) setShowFace(savedFace === 'true');
    const savedCtx = localStorage.getItem('bumble_gallery_show_ctx');
    if (savedCtx !== null) setShowContext(savedCtx === 'true');
    const savedKps = localStorage.getItem('bumble_gallery_show_kps');
    if (savedKps !== null) setShowLandmarks(savedKps === 'true');
    
    setMounted(true);
    fetch('/gallery_boxes.json')
      .then((r) => (r.ok ? r.json() : {}))
      .then((map) => setBoxesMap(map))
      .catch((e) => console.log('Notice: gallery_boxes.json load deferred', e));
  }, []);

  // Initial Fetch & Refetch when sortBy, setupFilter, or actionFilter changes (only after mounted and loaded localStorage states)
  useEffect(() => {
    if (!mounted) return;
    pageRef.current = 1;
    setPage(1);
    fetchProfiles(1, false, sortBy, false, setupFilter, actionFilter);
  }, [sortBy, setupFilter, actionFilter, fetchProfiles, mounted]);

  const loadMore = useCallback(() => {
    if (loading || fetchingRef.current || !hasMore) return;
    const nextPage = pageRef.current + 1;
    pageRef.current = nextPage;
    setPage(nextPage);
    fetchProfiles(nextPage, true, sortBy, false, setupFilter, actionFilter);
  }, [fetchProfiles, loading, hasMore, sortBy, setupFilter, actionFilter]);

  // Infinite Scroll IntersectionObserver Setup
  useEffect(() => {
    if (!hasMore || loading) return;

    const observer = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting) {
          loadMore();
        }
      },
      { threshold: 0.1 }
    );

    const target = observerTarget.current;
    if (target) {
      observer.observe(target);
    }

    return () => {
      if (target) {
        observer.unobserve(target);
      }
    };
  }, [hasMore, loading, loadMore]);

  // Set up real-time polling interval (only runs on latest sort view to prevent pagination score mixups)
  useEffect(() => {
    if (!isRealTime || sortBy !== 'latest') return;
    
    const interval = setInterval(() => {
      fetchProfiles(1, false, sortBy, true, setupFilter, actionFilter);
    }, 3000);
    
    return () => clearInterval(interval);
  }, [isRealTime, fetchProfiles, sortBy, setupFilter, actionFilter]);

  // Escape key event listener to close modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      if (e.key === "Escape") {
        setActiveProfile(null);
      } else if (e.key === "d" || e.key === "D") {
        setDissectMode((prev) => {
          const next = !prev;
          localStorage.setItem('bumble_gallery_dissect', String(next));
          return next;
        });
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const fetchingBoxesRef = useRef<Set<string>>(new Set());
  const getCardBoxData = (screenshotName: string): BoxData => {
    const cleanId = screenshotName.replace(/\.(webp|png|jpg|jpeg)$/i, '');
    if (boxesMap[cleanId]) return boxesMap[cleanId];
    if (boxesMap[screenshotName]) return boxesMap[screenshotName];

    // Trigger on-demand detection fetch if not already requested
    if (!fetchingBoxesRef.current.has(cleanId)) {
      fetchingBoxesRef.current.add(cleanId);
      fetch(`/api/dissect?file=${encodeURIComponent(screenshotName)}`)
        .then((r) => r.json())
        .then((res) => {
          if (res?.dissect) {
            setBoxesMap((prev) => ({ ...prev, [cleanId]: res.dissect }));
          }
        })
        .catch(() => {});
    }

    return {
      has_face: true,
      face_box: [0.38, 0.26, 0.62, 0.44],
      photo_box: [0.0, 0.078, 1.0, 0.60],
      ctx_box: [0.20, 0.16, 0.80, 0.54],
      kps: [
        [0.44, 0.32],
        [0.56, 0.32],
        [0.50, 0.37],
        [0.46, 0.41],
        [0.54, 0.41],
      ],
      det_score: 0.90,
    };
  };

  // Immediate detection fetch whenever a profile modal is opened
  useEffect(() => {
    if (!activeProfile?.screenshot) return;
    const cleanId = activeProfile.screenshot.replace(/\.(webp|png|jpg|jpeg)$/i, '');
    if (!boxesMap[cleanId]) {
      fetch(`/api/dissect?file=${encodeURIComponent(activeProfile.screenshot)}`)
        .then((r) => r.json())
        .then((res) => {
          if (res?.dissect) {
            setBoxesMap((prev) => ({ ...prev, [cleanId]: res.dissect }));
          }
        })
        .catch(() => {});
    }
  }, [activeProfile?.screenshot, boxesMap]);

  // Scroll listener to toggle compact/floating controls bar state
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 40);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleCopyText = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const handleSortChange = (newSort: 'latest' | 'oldest' | 'attractive_desc' | 'attractive_asc') => {
    setSortBy(newSort);
    localStorage.setItem('bumble_gallery_sort', newSort);
  };

  const handleRefresh = () => {
    pageRef.current = 1;
    setPage(1);
    fetchProfiles(1, false, sortBy, false, setupFilter, actionFilter);
  };

  return (
    <main className="relative min-h-screen bg-[#050505] text-white overflow-x-hidden selection:bg-purple-500/30 font-sans p-6 lg:p-10 flex flex-col">
      <div className="fixed top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-indigo-900/20 blur-[120px] pointer-events-none animate-pulse" style={{ animationDuration: '10s' }} />
      <div className="fixed bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-emerald-900/10 blur-[120px] pointer-events-none animate-pulse delay-1000" style={{ animationDuration: '7s' }} />

      <div className="relative z-10 w-full max-w-[1600px] mx-auto">
        <header className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 mb-10 pb-6 border-b border-white/10">
           <div>
             <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-xs font-semibold tracking-widest text-indigo-400 uppercase mb-4 shadow-[0_0_15px_rgba(99,102,241,0.1)]">
               Signal Intel Repository
             </div>
             <h1 className="text-4xl md:text-5xl font-black tracking-tighter bg-gradient-to-br from-white via-gray-200 to-gray-500 bg-clip-text text-transparent">
                Intel Gallery
             </h1>
           </div>
           <div className="flex gap-3">
             <Link href="/sandbox" className="bg-white/5 hover:bg-white/10 border border-white/10 transition-colors px-6 py-3 rounded-xl text-xs font-bold tracking-widest uppercase flex items-center gap-2">
               🧪 Profile Sandbox
             </Link>
             <Link href="/" className="bg-white/5 hover:bg-white/10 border border-white/10 transition-colors px-6 py-3 rounded-xl text-xs font-bold tracking-widest uppercase flex items-center gap-3">
               <span className="text-lg leading-none">←</span> Back to Matrix
             </Link>
           </div>
        </header>

        {/* Controls Bar */}
        <div className={`sticky z-50 transition-all duration-300 flex flex-row justify-between items-center gap-2 mb-8 backdrop-blur-xl ${
          isScrolled 
            ? 'top-[-8px] py-1.5 px-3 md:px-5 bg-black border-indigo-500/40 shadow-2xl shadow-indigo-500/10 rounded-xl border-b border-x' 
            : 'top-6 py-2.5 px-4 md:py-4 md:px-6 bg-[#0a0a0c]/95 border-white/20 shadow-lg rounded-2xl border'
        }`}>
          {/* Left Side: Stats */}
          <div className="text-[10px] md:text-xs font-bold tracking-wider text-white uppercase shrink-0">
            Loaded <span className="text-indigo-300 font-mono font-bold">{profiles.length}</span> of <span className="text-indigo-300 font-mono font-bold">{totalCount}</span> Profiles
          </div>

          {/* Right Side: Actions */}
          <div className="flex flex-row items-center gap-1.5 md:gap-3 justify-end overflow-x-auto no-scrollbar scroll-smooth shrink-1 min-w-0">
            
            {/* Setup filter dropdown */}
            <div className="flex items-center gap-1 bg-[#121215] border border-white/20 rounded-lg px-2 py-1 shrink-0">
              <span className="text-[9px] md:text-[10px] font-bold uppercase tracking-wider text-indigo-300">Setup:</span>
              <select
                id="gallery-setup-filter"
                value={setupFilter}
                onChange={(e) => {
                  const val = e.target.value;
                  setSetupFilter(val);
                  localStorage.setItem('bumble_gallery_setup_filter', val);
                }}
                className="bg-transparent text-[9px] md:text-[10px] font-bold uppercase tracking-wider text-white focus:text-white outline-none cursor-pointer border-none p-0 pr-5"
              >
                <option value="" className="bg-[#0f0f0f] text-white">All Setups</option>
                {availableSetups.map((setup) => (
                  <option key={setup} value={setup} className="bg-[#0f0f0f] text-white">
                    {setup}
                  </option>
                ))}
              </select>
            </div>

            {/* Decision filter dropdown */}
            <div className="flex items-center gap-1 bg-[#121215] border border-white/20 rounded-lg px-2 py-1 shrink-0">
              <span className="text-[9px] md:text-[10px] font-bold uppercase tracking-wider text-indigo-300">Decision:</span>
              <select
                id="gallery-action-filter"
                value={actionFilter}
                onChange={(e) => {
                  const val = e.target.value;
                  setActionFilter(val);
                  localStorage.setItem('bumble_gallery_action_filter', val);
                }}
                className="bg-transparent text-[9px] md:text-[10px] font-bold uppercase tracking-wider text-white focus:text-white outline-none cursor-pointer border-none p-0 pr-5"
              >
                <option value="" className="bg-[#0f0f0f] text-white">All Decisions</option>
                <option value="right" className="bg-[#0f0f0f] text-emerald-400">OK</option>
                <option value="left" className="bg-[#0f0f0f] text-rose-400">PASS</option>
              </select>
            </div>

            {/* Sort Buttons */}
            <div className="flex items-center gap-0.5 bg-[#121215] border border-white/20 rounded-lg p-0.5 shrink-0">
              {[
                { value: 'latest', label: 'Latest', id: 'gallery-sort-latest' },
                { value: 'oldest', label: 'Oldest', id: 'gallery-sort-oldest' },
                { value: 'attractive_desc', label: 'Attractive', id: 'gallery-sort-attractive-desc' },
                { value: 'attractive_asc', label: 'Unattractive', id: 'gallery-sort-attractive-asc' }
              ].map((opt) => (
                <button
                  key={opt.value}
                  id={opt.id}
                  onClick={() => handleSortChange(opt.value as any)}
                  className={`px-2 py-1 rounded-md text-[9px] md:text-[10px] font-bold uppercase tracking-wider transition-all duration-200 cursor-pointer ${
                    sortBy === opt.value
                      ? 'bg-indigo-500 text-white shadow-[0_0_12px_rgba(99,102,241,0.4)]'
                      : 'text-gray-200 hover:text-white hover:bg-white/10'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>

            {/* Dissect CV Toggle Button */}
            <div className="flex items-center gap-1 shrink-0">
              <button
                id="gallery-dissect-btn"
                onClick={() => {
                  const newVal = !dissectMode;
                  setDissectMode(newVal);
                  localStorage.setItem('bumble_gallery_dissect', String(newVal));
                }}
                className={`flex items-center gap-1.5 border transition-all px-2.5 py-1.5 rounded-lg text-[9px] md:text-[10px] font-bold tracking-widest uppercase cursor-pointer ${
                  dissectMode
                    ? 'bg-cyan-950/80 text-cyan-300 border-cyan-500/50 hover:bg-cyan-900 shadow-[0_0_15px_rgba(6,182,212,0.3)]'
                    : 'bg-[#121215] text-white border-white/20 hover:bg-white/10'
                }`}
                title="Toggle CV Dissection Overlays (Hotkey: D)"
              >
                <span>🔬</span>
                <span>Dissect {dissectMode ? 'ON' : 'OFF'}</span>
                <span className="text-[8px] opacity-60 font-mono">[D]</span>
              </button>

              {dissectMode && (
                <div className="flex items-center gap-0.5 bg-[#121215] border border-cyan-500/30 rounded-lg p-0.5 animate-fade-in">
                  <button
                    onClick={() => {
                      const v = !showFace;
                      setShowFace(v);
                      localStorage.setItem('bumble_gallery_show_face', String(v));
                    }}
                    className={`px-1.5 py-1 rounded text-[8px] font-bold uppercase transition-all ${
                      showFace ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-500/40' : 'text-gray-400 hover:text-white'
                    }`}
                    title="Toggle Face Biometrics Bounding Box"
                  >
                    Face
                  </button>
                  <button
                    onClick={() => {
                      const v = !showContext;
                      setShowContext(v);
                      localStorage.setItem('bumble_gallery_show_ctx', String(v));
                    }}
                    className={`px-1.5 py-1 rounded text-[8px] font-bold uppercase transition-all ${
                      showContext ? 'bg-amber-500/30 text-amber-200 border border-amber-500/40' : 'text-gray-400 hover:text-white'
                    }`}
                    title="Toggle SigLIP Scene Context Boundary"
                  >
                    Context
                  </button>
                  <button
                    onClick={() => {
                      const v = !showLandmarks;
                      setShowLandmarks(v);
                      localStorage.setItem('bumble_gallery_show_kps', String(v));
                    }}
                    className={`px-1.5 py-1 rounded text-[8px] font-bold uppercase transition-all ${
                      showLandmarks ? 'bg-emerald-500/30 text-emerald-200 border border-emerald-500/40' : 'text-gray-400 hover:text-white'
                    }`}
                    title="Toggle 5-Point Facial Landmarks"
                  >
                    Landmarks
                  </button>
                </div>
              )}
            </div>

            {/* Real-time Toggle Button */}
            <button
              id="gallery-realtime-btn"
              onClick={() => {
                const newVal = !isRealTime;
                setIsRealTime(newVal);
                localStorage.setItem('bumble_gallery_realtime', String(newVal));
              }}
              className={`flex items-center gap-1.5 border transition-all px-2.5 py-1.5 rounded-lg text-[9px] md:text-[10px] font-bold tracking-widest uppercase cursor-pointer shrink-0 ${
                isRealTime
                  ? 'bg-emerald-950/80 text-emerald-400 border-emerald-500/40 hover:bg-emerald-900 shadow-[0_0_15px_rgba(16,185,129,0.2)]'
                  : 'bg-[#121215] text-white border-white/20 hover:bg-white/10'
              }`}
            >
              <span className="relative flex h-1.5 w-1.5">
                {isRealTime && (
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                )}
                <span className={`relative inline-flex rounded-full h-1.5 w-1.5 ${isRealTime ? 'bg-emerald-500' : 'bg-gray-500'}`}></span>
              </span>
              {isRealTime ? "Active" : "Live"}
            </button>

            {/* Refresh Button */}
            <button
              id="gallery-refresh-btn"
              onClick={handleRefresh}
              disabled={loading}
              className="flex items-center gap-1.5 bg-[#121215] hover:bg-indigo-500/20 text-white hover:text-indigo-300 border border-white/20 hover:border-indigo-500/35 transition-all px-2.5 py-1.5 rounded-lg text-[9px] md:text-[10px] font-bold tracking-widest uppercase cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed group shrink-0"
            >
              <svg 
                className={`w-3 h-3 fill-current transition-transform duration-500 ${loading ? 'animate-spin' : 'group-hover:rotate-180'}`} 
                viewBox="0 0 24 24"
              >
                <path d="M19 12a7 7 0 0 0-7-7c-1.86 0-3.55.73-4.8 1.91L9 9H3V3l2.25 2.25C6.9 3.63 9.3 2.5 12 2.5a9.5 9.5 0 0 1 9.5 9.5H19m-7 7c1.86 0 3.55-.73 4.8-1.91L15 15h6v6l-2.25-2.25C17.1 20.37 14.7 21.5 12 21.5A9.5 9.5 0 0 1 2.5 12H5a7 7 0 0 0 7 7z"/>
              </svg>
              Refresh
            </button>

          </div>
        </div>

        {profiles.length === 0 && loading ? (
          <div className="flex justify-center p-20">
             <div className="w-12 h-12 border-4 border-white/10 border-t-indigo-500 rounded-full animate-spin"></div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
                {profiles.map((p, idx) => (
                  <div 
                    key={p.id} 
                    onClick={() => {
                      setActiveProfile(p);
                      setCopiedKey(null);
                    }}
                    className={`group relative bg-white/5 rounded-2xl overflow-hidden border ${
                      dissectMode
                        ? (p.action === 'right' ? 'border-cyan-500/60 shadow-[0_0_15px_rgba(6,182,212,0.25)] ring-1 ring-cyan-500/30' : 'border-slate-800 hover:border-cyan-500/40')
                        : 'border-white/10 hover:border-indigo-500/50 shadow-xl hover:shadow-[0_0_20px_rgba(99,102,241,0.2)]'
                    } aspect-[720/1480] flex flex-col cursor-pointer transition-all duration-300 hover:-translate-y-1 animate-fade-in`}
                  >
                     {/* eslint-disable-next-line @next/next/no-img-element */}
                     <img 
                       src={`/api/image?file=${encodeURIComponent(p.screenshot)}`} 
                       alt={`Profile ${p.id}`}
                       loading="lazy"
                       className="absolute inset-0 w-full h-full object-cover object-center transition-transform duration-700 group-hover:scale-105"
                     />
                     
                     {/* Dissect Overlay for Thumbnails */}
                     {dissectMode && (
                       <DissectOverlay
                         boxData={getCardBoxData(p.screenshot)}
                         v6r5Score={p.v6r5_z ?? p.face_biased}
                         loraScore={p.lora_z ?? p.multimodal}
                         finalScore={p.score}
                         action={p.action}
                         variant="thumbnail"
                         showFace={showFace}
                         showContext={showContext}
                         showLandmarks={showLandmarks}
                       />
                     )}
                     
                     {/* Rank Badge for Attractiveness Sorting */}
                     {(sortBy === 'attractive_desc' || sortBy === 'attractive_asc') && (
                       <span className="absolute top-5 left-5 z-10 text-4xl font-black font-mono text-white tracking-tighter drop-shadow-xl select-none animate-fade-in">
                         #{idx + 1}
                       </span>
                     )}                     <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
                     <div className="absolute bottom-0 inset-x-0 p-5 flex justify-between items-end">
                        <div className="flex flex-col items-start">
                           <span className="text-4xl font-black font-mono text-white tracking-tighter drop-shadow-xl">
                             {p.score == null ? '--' : p.score.toFixed(1)}
                             {p.decision_mode === 'preference' && '%'}
                           </span>
                         </div>
                        <span className={`text-2xl font-black italic uppercase tracking-tighter transform -skew-x-6 drop-shadow-lg ${p.action === 'right' ? 'text-emerald-400' : 'text-rose-500'}`}>
                          {p.action === 'right' ? 'OK' : 'PASS'}
                        </span>
                     </div>
                  </div>
                ))}
            </div>

            {/* Observer Target / Loading Indicator */}
            {hasMore && (
              <div ref={observerTarget} className="mt-12 flex justify-center py-8">
                 <div className="w-10 h-10 border-4 border-white/10 border-t-indigo-500 rounded-full animate-spin"></div>
              </div>
            )}

            {!hasMore && profiles.length > 0 && (
              <div className="mt-16 text-center text-gray-500 text-xs tracking-widest uppercase font-bold">
                 End of Repository
              </div>
            )}
          </>
        )}
      </div>      {/* Enlarged Telemetry Preview Modal Overlay */}
      {activeProfile && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-10 select-none">
          {/* Backdrop blur click to close */}
          <div 
            className="absolute inset-0 bg-[#050505]/95 backdrop-blur-md cursor-pointer transition-opacity duration-300"
            onClick={() => setActiveProfile(null)}
          />
          
          {/* Modal Container */}
          <div className="relative z-10 w-full max-w-6xl bg-[#0c0c0e]/90 border border-white/10 rounded-3xl overflow-hidden shadow-2xl flex flex-col md:flex-row max-h-[90vh] md:max-h-[80vh] animate-in fade-in zoom-in duration-200">
            
            {/* Left Side: Enlarged Screenshot */}
            <div className="relative flex-1 bg-black/60 flex flex-col items-center justify-center p-4 min-h-[300px] md:min-h-0 overflow-hidden">
              <div 
                className="relative inline-block rounded-xl overflow-hidden shadow-2xl"
                style={{ aspectRatio: '720 / 1480', maxHeight: '72vh', maxWidth: '100%' }}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img 
                  src={`/api/image?file=${encodeURIComponent(activeProfile.screenshot)}`} 
                  alt="Profile Preview" 
                  className="w-full h-full object-cover select-none block"
                />
                {dissectMode && (
                  <DissectOverlay
                    boxData={getCardBoxData(activeProfile.screenshot)}
                    v6r5Score={activeProfile.v6r5_z ?? activeProfile.face_biased}
                    loraScore={activeProfile.lora_z ?? activeProfile.multimodal}
                    finalScore={activeProfile.score}
                    action={activeProfile.action}
                    variant="modal"
                    showFace={showFace}
                    showContext={showContext}
                    showLandmarks={showLandmarks}
                  />
                )}
              </div>
              
              {/* Modal Dissect Toolbar */}
              <div className="mt-3 flex items-center gap-2 bg-[#0a0c10]/90 border border-white/10 rounded-xl px-3 py-1.5 backdrop-blur-md z-30">
                <button
                  onClick={() => setDissectMode(!dissectMode)}
                  className={`px-2.5 py-1 rounded-lg text-[9px] font-bold uppercase transition-all flex items-center gap-1 cursor-pointer ${
                    dissectMode ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-500/40' : 'bg-white/5 text-gray-400 border border-white/10'
                  }`}
                >
                  <span>🔬</span>
                  <span>Dissect {dissectMode ? 'ON' : 'OFF'}</span>
                </button>
                {dissectMode && (
                  <>
                    <button
                      onClick={() => setShowFace(!showFace)}
                      className={`px-2 py-0.5 rounded text-[8px] font-bold uppercase transition-all cursor-pointer ${
                        showFace ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-500/40' : 'text-gray-400 hover:text-white'
                      }`}
                    >
                      Face Box
                    </button>
                    <button
                      onClick={() => setShowContext(!showContext)}
                      className={`px-2 py-0.5 rounded text-[8px] font-bold uppercase transition-all cursor-pointer ${
                        showContext ? 'bg-amber-500/30 text-amber-200 border border-amber-500/40' : 'text-gray-400 hover:text-white'
                      }`}
                    >
                      Context Box
                    </button>
                    <button
                      onClick={() => setShowLandmarks(!showLandmarks)}
                      className={`px-2 py-0.5 rounded text-[8px] font-bold uppercase transition-all cursor-pointer ${
                        showLandmarks ? 'bg-emerald-500/30 text-emerald-200 border border-emerald-500/40' : 'text-gray-400 hover:text-white'
                      }`}
                    >
                      Landmarks
                    </button>
                  </>
                )}
              </div>
            </div>
 
            {/* Right Side: Telemetry details */}
            <div className="w-full md:w-[460px] shrink-0 bg-[#070708] border-t md:border-t-0 md:border-l border-white/10 p-6 flex flex-col justify-between overflow-y-auto">
              <div className="flex flex-col gap-6">
                
                {/* Header / Title */}
                <div className="flex justify-between items-start gap-4">
                  <div>
                    <h3 className="text-gray-400 font-medium text-xs tracking-widest uppercase mb-1">Signal Meta</h3>
                    <h2 className="text-xl font-black text-white tracking-tight">Telemetry Breakdown</h2>
                  </div>
                  <button 
                    onClick={() => setActiveProfile(null)}
                    className="w-8 h-8 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 flex items-center justify-center text-gray-400 hover:text-white transition-all text-sm cursor-pointer"
                  >
                    ✕
                  </button>
                </div>
 
                {/* Score Showcase */}
                <div className="relative bg-gradient-to-br from-indigo-500/10 to-purple-500/10 rounded-2xl p-5 border border-indigo-500/20 flex items-center justify-between shadow-inner">
                  <div className="flex flex-col">
                    <span className="text-[10px] text-indigo-300 font-semibold uppercase tracking-wider mb-1">
                      {activeProfile.method?.startsWith('v6') ? 'Gambler Score' : (activeProfile.decision_mode === 'preference' ? 'Spline Preference Probability' : 'Blended Final Score')}
                    </span>
                    <span className="text-5xl font-black font-mono text-white tracking-tighter">
                      {activeProfile.score == null ? '--' : activeProfile.score.toFixed(1)}
                      {activeProfile.decision_mode === 'preference' && '%'}
                    </span>
                  </div>
                  <div className="flex flex-col items-end">
                    <span className="text-[9px] text-gray-400 uppercase tracking-widest mb-1">Decision</span>
                    <span className={`text-2xl font-black italic uppercase tracking-tighter transform -skew-x-6 ${activeProfile.action === 'right' ? 'text-emerald-400 drop-shadow-[0_0_10px_rgba(16,185,129,0.3)]' : 'text-rose-500 drop-shadow-[0_0_10px_rgba(244,63,94,0.3)]'}`}>
                      {activeProfile.action === 'right' ? 'OK' : 'PASS'}
                    </span>
                  </div>
                </div>

                {/* Sub-scores metrics grid */}
                <div className="flex flex-col gap-4">
                  <h4 className="text-gray-500 font-medium text-[10px] tracking-widest uppercase">Regressor Output Breakdown</h4>
                  <div className="grid grid-cols-2 gap-3">
                    {(activeProfile.method?.startsWith('v6') ? [
                      { name: "v6r5 Z", value: activeProfile.v6r5_z ?? activeProfile.face_biased, color: "text-purple-400" },
                      { name: "LoRA Z", value: activeProfile.lora_z ?? activeProfile.multimodal, color: "text-blue-400" },
                      { name: "Agreement", value: activeProfile.psi_sync ?? activeProfile.ridge, color: "text-amber-400" },
                      { name: "Gambler", value: activeProfile.gambler_score ?? activeProfile.knn, color: "text-pink-400" }
                    ] : [
                      { name: "Face Biased", value: activeProfile.face_biased, color: "text-purple-400" },
                      { name: "Multimodal", value: activeProfile.multimodal, color: "text-blue-400" },
                      { name: "Ridge Regressor", value: activeProfile.ridge, color: "text-amber-400" },
                      { name: "k-NN Similarity", value: activeProfile.knn, color: "text-pink-400" }
                    ]).map((metric) => (
                      <div key={metric.name} className="bg-white/5 border border-white/5 rounded-xl p-3.5 flex flex-col gap-1 transition-colors hover:bg-white/10">
                        <span className="text-gray-400 text-[9px] font-bold uppercase tracking-wider">{metric.name}</span>
                        <span className={`text-lg font-black font-mono tracking-tight ${metric.color}`}>
                          {metric.value !== null && metric.value !== undefined ? metric.value.toFixed(2) : "--"}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Computer Vision Dissection Telemetry Panel */}
                <div className="bg-[#0b0e14] border border-cyan-500/25 rounded-2xl p-4 flex flex-col gap-3 shadow-[0_0_15px_rgba(6,182,212,0.06)]">
                  <div className="flex justify-between items-center pb-2 border-b border-cyan-500/20">
                    <span className="text-cyan-300 text-[10px] font-bold uppercase tracking-wider flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                      CV Dissection Telemetry
                    </span>
                    <span className="px-2 py-0.5 rounded text-[8px] font-black uppercase tracking-wider bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                      v6.8 Architecture
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="bg-white/5 border border-amber-500/20 rounded-xl p-2.5">
                      <span className="text-amber-400 text-[8.5px] font-bold uppercase tracking-wider block mb-0.5">SigLIP SO400M Context</span>
                      <span className="font-mono text-white text-sm font-black">
                        {activeProfile.lora_z !== null && activeProfile.lora_z !== undefined ? `${activeProfile.lora_z >= 0 ? '+' : ''}${activeProfile.lora_z.toFixed(2)}z` : (activeProfile.multimodal !== null ? activeProfile.multimodal?.toFixed(2) : '--')}
                      </span>
                      <div className="text-[8px] text-gray-400 font-mono mt-0.5 flex justify-between">
                        <span>Weight: 70%</span>
                        <span className="text-amber-300/80">Crop: y∈[7.8%, 60%]</span>
                      </div>
                    </div>

                    <div className="bg-white/5 border border-cyan-500/20 rounded-xl p-2.5">
                      <span className="text-cyan-400 text-[8.5px] font-bold uppercase tracking-wider block mb-0.5">ArcFace 512D + DINOv3</span>
                      <span className="font-mono text-white text-sm font-black">
                        {activeProfile.v6r5_z !== null && activeProfile.v6r5_z !== undefined ? `${activeProfile.v6r5_z >= 0 ? '+' : ''}${activeProfile.v6r5_z.toFixed(2)}z` : (activeProfile.face_biased !== null ? activeProfile.face_biased?.toFixed(2) : '--')}
                      </span>
                      <div className="text-[8px] text-gray-400 font-mono mt-0.5 flex justify-between">
                        <span>Weight: 30%</span>
                        <span className="text-cyan-300/80">5-Pt Landmarks</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex justify-between items-center bg-black/40 border border-white/5 rounded-xl p-2.5 text-[9px] font-mono">
                    <span className="text-gray-400">Blended Inference:</span>
                    <span className="font-black text-white">
                      0.30*(v6r5) + 0.70*(LoRA) = <b className={activeProfile.action === 'right' ? 'text-emerald-400' : 'text-rose-400'}>{activeProfile.score == null ? '--' : activeProfile.score.toFixed(2)}</b>
                    </span>
                  </div>
                </div>

                {/* Strategy & Run Config */}
                <div className="bg-white/5 border border-white/5 rounded-2xl p-4 flex flex-col gap-3">
                  <div className="flex justify-between items-center pb-2 border-b border-white/5">
                    <span className="text-gray-400 text-[10px] font-bold uppercase tracking-wider">Strategy & Run Config</span>
                    <span className="px-2.5 py-0.5 rounded-full text-[9px] font-black uppercase tracking-wider bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                      {activeProfile.setup_name || 'Primitive'}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-3 text-xs">
                    <div>
                      <span className="text-gray-500 text-[9px] font-bold uppercase tracking-widest block mb-0.5">Decision Mode</span>
                      <span className="font-semibold text-white capitalize">{activeProfile.decision_mode || 'threshold'}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 text-[9px] font-bold uppercase tracking-widest block mb-0.5">Run Provider</span>
                      <span className="font-semibold text-white uppercase font-mono">{activeProfile.provider || 'cpu'}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 text-[9px] font-bold uppercase tracking-widest block mb-0.5">Core Method</span>
                      <span className="font-semibold text-gray-300 font-mono">{activeProfile.method || 'N/A'}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 text-[9px] font-bold uppercase tracking-widest block mb-0.5">{activeProfile.method?.startsWith('v6') ? 'v6r5 Weight' : 'Face Weight'}</span>
                      <span className="font-semibold text-gray-300 font-mono">{activeProfile.method?.startsWith('v6') ? '0.30' : ((activeProfile.face_weight !== undefined && activeProfile.face_weight !== null) ? activeProfile.face_weight.toFixed(2) : 'N/A')}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 text-[9px] font-bold uppercase tracking-widest block mb-0.5">{activeProfile.method?.startsWith('v6') ? 'LoRA Weight' : 'k-NN Neighbors'}</span>
                      <span className="font-semibold text-gray-300 font-mono">{activeProfile.method?.startsWith('v6') ? '0.70' : (activeProfile.k ?? 'N/A')}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 text-[9px] font-bold uppercase tracking-widest block mb-0.5">24/7 Run Mode</span>
                      <span className={`font-semibold font-mono text-[10px] ${activeProfile.mode_247 ? 'text-emerald-400' : 'text-gray-500'}`}>{activeProfile.mode_247 ? 'ACTIVE' : 'INACTIVE'}</span>
                    </div>
                  </div>
                </div>

                {/* Threshold & Swiping Logic */}
                <div className="bg-white/5 border border-white/5 rounded-2xl p-4 flex flex-col gap-3">
                  <div className="flex justify-between items-center pb-2 border-b border-white/5">
                    <span className="text-gray-400 text-[10px] font-bold uppercase tracking-wider">Threshold & Swiping Logic</span>
                    <span className="px-2.5 py-0.5 rounded-full text-[9px] font-black uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/30">
                      {activeProfile.decision_mode === 'preference' 
                        ? (activeProfile.dynamic_preference_enabled ? 'Dynamic Spline Cutoff' : 'Static Spline Cutoff')
                        : (activeProfile.dynamic_enabled ? 'Dynamic Cutoff' : 'Static Cutoff')}
                    </span>
                  </div>
                  
                  {activeProfile.decision_mode === 'preference' ? (
                    <div className="flex flex-col gap-2.5">
                      <div className="flex justify-between items-center">
                        <div>
                          <span className="text-gray-500 text-[9px] font-bold uppercase tracking-widest block mb-0.5">Spline Prob P(like)</span>
                          <span className="font-black text-lg text-indigo-300 font-mono">
                            {activeProfile.preference_probability !== undefined && activeProfile.preference_probability !== null 
                              ? `${(activeProfile.preference_probability * 100).toFixed(2)}%` 
                              : 'N/A'}
                          </span>
                        </div>
                        <div className="text-right">
                          <span className="text-gray-500 text-[9px] font-bold uppercase tracking-widest block mb-0.5">
                            {activeProfile.dynamic_preference_enabled ? 'Dynamic Cutoff' : 'Static Cutoff'}
                          </span>
                          <span className="font-black text-lg text-amber-400 font-mono">
                            {activeProfile.preference_threshold !== undefined && activeProfile.preference_threshold !== null 
                              ? `${(activeProfile.preference_threshold * 100).toFixed(2)}%` 
                              : 'N/A'}
                          </span>
                        </div>
                      </div>
                      
                      {/* Visual meter */}
                      {activeProfile.preference_probability !== undefined && activeProfile.preference_probability !== null && activeProfile.preference_threshold !== undefined && activeProfile.preference_threshold !== null && (
                        <div className="w-full flex flex-col gap-1 mt-1">
                          <div className="h-2 w-full bg-white/10 rounded-full overflow-hidden relative">
                            <div 
                              className="absolute top-0 bottom-0 left-0 bg-indigo-500 rounded-full" 
                              style={{ width: `${Math.min(100, activeProfile.preference_probability * 100)}%` }}
                            />
                            <div 
                              className="absolute top-0 bottom-0 w-0.5 bg-amber-400 z-10" 
                              style={{ left: `${activeProfile.preference_threshold * 100}%` }}
                              title="Threshold Cutoff"
                            />
                          </div>
                          <div className="flex justify-between text-[8px] text-gray-500 font-mono font-bold uppercase">
                            <span>0%</span>
                            <span className="text-amber-400">
                              Cutoff: {(activeProfile.preference_threshold * 100).toFixed(1)}%
                            </span>
                            <span>100%</span>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="flex flex-col gap-2">
                      <div className="flex justify-between items-center">
                        <div>
                          <span className="text-gray-500 text-[9px] font-bold uppercase tracking-widest block mb-0.5">Final Score</span>
                          <span className="font-black text-lg text-indigo-300 font-mono">
                            {activeProfile.score == null ? '--' : activeProfile.score.toFixed(2)}
                          </span>
                        </div>
                        <div className="text-right">
                          <span className="text-gray-500 text-[9px] font-bold uppercase tracking-widest block mb-0.5">
                            {activeProfile.dynamic_enabled ? 'Dynamic Cutoff' : 'Static Cutoff'}
                          </span>
                          <span className="font-black text-lg text-amber-400 font-mono">
                            {activeProfile.threshold !== undefined && activeProfile.threshold !== null 
                              ? activeProfile.threshold.toFixed(2) 
                              : 'N/A'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Dynamic Optimization Stats */}
                {((activeProfile.decision_mode === 'preference' && activeProfile.dynamic_preference_enabled) || 
                  (activeProfile.decision_mode !== 'preference' && activeProfile.dynamic_enabled)) && (
                  <div className="bg-emerald-950/20 border border-emerald-500/25 rounded-2xl p-4 flex flex-col gap-3 shadow-[0_0_15px_rgba(16,185,129,0.05)]">
                    <div className="flex justify-between items-center pb-2 border-b border-emerald-500/15">
                      <span className="text-emerald-400 text-[10px] font-bold uppercase tracking-wider flex items-center gap-1.5">
                        <span className="relative flex h-1.5 w-1.5">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500"></span>
                        </span>
                        Dynamic Auto-Tuning Active
                      </span>
                      <span className="px-2 py-0.5 rounded text-[8px] font-black uppercase tracking-wider bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                        Feedback Loop
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-x-4 gap-y-3 text-xs">
                      <div>
                        <span className="text-emerald-500/60 text-[9px] font-bold uppercase tracking-widest block mb-0.5">Target Swipe Rate</span>
                        <span className="font-semibold text-emerald-300 font-mono">
                          {activeProfile.decision_mode === 'preference' 
                            ? `${((activeProfile.dynamic_preference_target_right_rate ?? 0.2) * 100).toFixed(0)}%`
                            : `${((activeProfile.dynamic_target_right_rate ?? 0.2) * 100).toFixed(0)}%`}
                        </span>
                      </div>
                      <div>
                        <span className="text-emerald-500/60 text-[9px] font-bold uppercase tracking-widest block mb-0.5">Sliding Window</span>
                        <span className="font-semibold text-emerald-300 font-mono">
                          {activeProfile.decision_mode === 'preference' 
                            ? activeProfile.dynamic_preference_window 
                            : activeProfile.dynamic_window} profiles
                        </span>
                      </div>
                      <div>
                        <span className="text-emerald-500/60 text-[9px] font-bold uppercase tracking-widest block mb-0.5">Min History</span>
                        <span className="font-semibold text-emerald-300 font-mono">
                          {activeProfile.decision_mode === 'preference' 
                            ? activeProfile.dynamic_preference_min_history 
                            : activeProfile.dynamic_min_history}
                        </span>
                      </div>
                      <div>
                        <span className="text-emerald-500/60 text-[9px] font-bold uppercase tracking-widest block mb-0.5">Tuning Percentile</span>
                        <span className="font-semibold text-emerald-300 font-mono">
                          {activeProfile.decision_mode === 'preference' 
                            ? `${activeProfile.dynamic_preference_percentile ?? 80}th` 
                            : `${activeProfile.dynamic_percentile ?? 80}th`}
                        </span>
                      </div>
                      <div>
                        <span className="text-emerald-500/60 text-[9px] font-bold uppercase tracking-widest block mb-0.5">Tuning Mode</span>
                        <span className="font-semibold text-emerald-300 font-mono uppercase">
                          {activeProfile.decision_mode === 'preference' 
                            ? (activeProfile.dynamic_preference_mode || 'N/A') 
                            : (activeProfile.dynamic_mode || 'N/A')}
                        </span>
                      </div>
                      <div>
                        <span className="text-emerald-500/60 text-[9px] font-bold uppercase tracking-widest block mb-0.5">Tuned Range</span>
                        <span className="font-semibold text-emerald-300 font-mono block truncate">
                          {activeProfile.decision_mode === 'preference' 
                            ? `${activeProfile.dynamic_preference_min_threshold !== null && activeProfile.dynamic_preference_min_threshold !== undefined ? `${(activeProfile.dynamic_preference_min_threshold * 100).toFixed(0)}%` : '0%'} - ${activeProfile.dynamic_preference_max_threshold !== null && activeProfile.dynamic_preference_max_threshold !== undefined ? `${(activeProfile.dynamic_preference_max_threshold * 100).toFixed(0)}%` : '100%'}`
                            : `${activeProfile.dynamic_min_threshold ?? 0}% - ${activeProfile.dynamic_max_threshold ?? 100}%`}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* System Paths & Model Artifacts */}
                <div className="bg-white/5 border border-white/5 rounded-2xl p-4 flex flex-col gap-3">
                  <span className="text-gray-400 text-[10px] font-bold uppercase tracking-wider pb-2 border-b border-white/5 block">System Paths & Model Artifacts</span>
                  
                  <div className="flex flex-col gap-3">
                    {[
                      { label: "Screenshot File", value: activeProfile.screenshot, key: "screenshot" },
                      { label: "Preference Veto Model", value: activeProfile.preference_model_path, key: "preference_model" },
                      { label: "Rating Regressor", value: activeProfile.regressor_path, key: "regressor" },
                      { label: "Multimodal Regressor", value: activeProfile.multimodal_regressor_path, key: "multimodal_regressor" },
                      { label: "Reference Embeddings Store", value: activeProfile.store_path, key: "store" }
                    ].map((item) => {
                      if (!item.value) return null;
                      const isCopied = copiedKey === item.key;
                      return (
                        <div key={item.key} className="flex flex-col gap-1.5 bg-black/40 border border-white/5 rounded-xl p-3 overflow-hidden">
                          <span className="text-[8px] text-gray-500 font-bold uppercase tracking-widest block">{item.label}</span>
                          <div className="text-[9px] font-mono text-gray-300 break-all select-all leading-relaxed max-h-[60px] overflow-y-auto">
                            {item.value}
                          </div>
                          <button 
                            onClick={() => handleCopyText(item.value!, item.key)}
                            className={`self-start inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border transition-all text-[8px] font-bold uppercase tracking-widest cursor-pointer ${
                              isCopied 
                                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' 
                                : 'bg-white/5 text-gray-400 border-white/10 hover:bg-indigo-500/20 hover:text-indigo-300'
                            }`}
                          >
                            {isCopied ? "✓ Copied!" : "📄 Copy Path"}
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
                
              </div>
 
              <div className="mt-8 flex flex-col gap-2.5">
                <div className="text-[9px] font-semibold text-gray-500 uppercase tracking-widest font-mono text-center">
                  Timestamp: {activeProfile.timestamp || "N/A"}
                </div>
                <button 
                  onClick={() => setActiveProfile(null)}
                  className="w-full py-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-xs font-bold uppercase tracking-widest transition-all cursor-pointer text-center"
                >
                  Dismiss Preview
                </button>
              </div>
              
            </div>
            
          </div>
        </div>
      )}
    </main>
  );
}
