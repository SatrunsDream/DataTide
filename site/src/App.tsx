import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Search,
  Map as MapIcon,
  ChevronLeft,
  Info,
  Share2,
  Droplets,
  Wind,
  Thermometer,
  Star,
  Sun,
  Waves,
  Wind as WindIcon,
  Moon,
  Activity,
  Users,
} from 'lucide-react';
import { loadSurfSpots } from './services/surfSpots';
import type { SurfSpot } from './types';
import { CoastalMap } from './components/CoastalMap';
import { DataSurfLogo } from './components/Visuals';
import { AboutPage } from './components/AboutPage';
import {
  fetchBeachWeather,
  calculateSurfScore,
  uvIndexLabel,
  formatWindRelationLabel,
  type SurfConditions,
} from './services/weatherService';

const OPEN_METEO = 'https://open-meteo.com/';

type AppView = 'map' | 'search' | 'saved' | 'about';

type ThemeMode = 'system' | 'light' | 'dark';

const WaveBackground = () => (
  <div className="absolute inset-0 z-[-1] overflow-hidden pointer-events-none select-none">
    <div className="absolute inset-0 bg-gradient-to-b from-sand via-seafoam/20 to-ocean-main/5" />
    <div className="absolute top-0 left-0 w-[400%] h-full animate-wave-drift-slow opacity-[0.05]">
      <div className="absolute bottom-0 left-0 w-full h-[600px] wave-bg-pattern scale-y-50" />
    </div>
    <div className="absolute top-0 left-0 w-[400%] h-full animate-wave-drift-fast opacity-[0.03]">
      <div className="absolute bottom-[-50px] left-0 w-full h-[700px] wave-bg-pattern scale-y-75 flip-v" />
    </div>
    <div className="absolute bottom-0 left-0 right-0 h-[30vh] bg-gradient-to-t from-white/40 to-transparent" />
  </div>
);

function SpotListRow({
  spot,
  onSelect,
  favorites,
}: {
  spot: SurfSpot;
  onSelect: (s: SurfSpot) => void;
  favorites: string[];
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(spot)}
      className="w-full flex items-center gap-4 text-left p-4 rounded-2xl hover:bg-white/80 dark:hover:bg-white/5 transition-colors border border-transparent hover:border-ocean-main/10 group"
    >
      <div className="w-10 h-10 rounded-xl bg-ocean-main/10 flex items-center justify-center flex-shrink-0 text-ocean-main">
        <Waves size={18} aria-hidden />
      </div>
      <div className="flex-1 min-w-0">
        <h3 className="text-base font-serif truncate group-hover:text-ocean-main transition-colors">{spot.name}</h3>
        <p className="text-xs text-slate-500 font-medium dark:text-slate-400">{spot.county}</p>
      </div>
      {favorites.includes(spot.id) && <Star size={14} fill="#FFD23F" className="text-sun shrink-0" aria-hidden />}
    </button>
  );
}

function formatDayLabel(isoDate: string): string {
  try {
    const d = new Date(isoDate + 'T12:00:00');
    return d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
  } catch {
    return isoDate;
  }
}

export default function App() {
  const [spots, setSpots] = useState<SurfSpot[]>([]);

  useEffect(() => {
    loadSurfSpots().then(setSpots);
  }, []);

  const [selectedSpot, setSelectedSpot] = useState<SurfSpot | null>(null);
  const [view, setView] = useState<AppView>('map');
  const [searchQuery, setSearchQuery] = useState('');
  const [weather, setWeather] = useState<SurfConditions | null>(null);
  const [loadingWeather, setLoadingWeather] = useState(false);
  const [weatherError, setWeatherError] = useState(false);
  const [resumeSpotAfterAbout, setResumeSpotAfterAbout] = useState<SurfSpot | null>(null);
  const [themeMode, setThemeMode] = useState<ThemeMode>('system');

  const [favorites, setFavorites] = useState<string[]>(() => {
    const saved = localStorage.getItem('datasurf_favorites');
    return saved ? JSON.parse(saved) : [];
  });
  const [userProfile, setUserProfile] = useState<'surfer' | 'swimmer' | 'family'>('surfer');

  useEffect(() => {
    localStorage.setItem('datasurf_favorites', JSON.stringify(favorites));
  }, [favorites]);

  useEffect(() => {
    const apply = () => {
      let dark = false;
      if (themeMode === 'dark') dark = true;
      else if (themeMode === 'light') dark = false;
      else dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
    };
    apply();
    if (themeMode !== 'system') return;
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener('change', apply);
    return () => mq.removeEventListener('change', apply);
  }, [themeMode]);

  useEffect(() => {
    if (!selectedSpot) {
      setWeather(null);
      setWeatherError(false);
      setLoadingWeather(false);
      return;
    }
    setWeatherError(false);
    setLoadingWeather(true);
    fetchBeachWeather(selectedSpot.lat, selectedSpot.lng, selectedSpot.shorelineBearing ?? 270)
      .then((data) => {
        setWeather(data);
        setWeatherError(false);
      })
      .catch(() => {
        setWeather(null);
        setWeatherError(true);
      })
      .finally(() => setLoadingWeather(false));
  }, [selectedSpot]);

  const toggleFavorite = (id: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    setFavorites((prev) => (prev.includes(id) ? prev.filter((f) => f !== id) : [...prev, id]));
  };

  const q = searchQuery.toLowerCase().trim();
  const filteredSpots = spots.filter(
    (s) => !q || s.name.toLowerCase().includes(q) || s.county.toLowerCase().includes(q),
  );

  const savedSpots = spots.filter((s) => favorites.includes(s.id));

  const surfScoreRounded =
    selectedSpot && weather ? Math.round(calculateSurfScore(weather, userProfile)) : null;

  const profileScoreLabel =
    userProfile === 'surfer' ? 'Surfer score' : userProfile === 'swimmer' ? 'Swimmer score' : 'Family score';

  const handleBack = () => setSelectedSpot(null);
  const handleShare = () => {
    if (selectedSpot && navigator.share) {
      navigator.share({
        title: `DataSurf | ${selectedSpot.name}`,
        text: `Surf conditions at ${selectedSpot.name} (${selectedSpot.county})`,
        url: window.location.href,
      });
    }
  };

  const openAbout = () => {
    if (selectedSpot) setResumeSpotAfterAbout(selectedSpot);
    setView('about');
  };

  const closeAbout = () => {
    setView('map');
    if (resumeSpotAfterAbout) {
      setSelectedSpot(resumeSpotAfterAbout);
      setResumeSpotAfterAbout(null);
    }
  };

  const mapDimmed = view === 'search' || view === 'saved';

  const bottomSheetSpots = useMemo(() => spots.slice(0, 12), [spots]);

  return (
    <div className="relative h-screen w-full overflow-hidden bg-sand font-sans select-none">
      <WaveBackground />

      <header className="fixed top-0 left-0 right-0 min-h-24 px-4 sm:px-8 z-50 flex flex-wrap items-center justify-between gap-2 pointer-events-none py-3">
        <div className="pointer-events-auto">
          <div
            onClick={() => {
              setView('map');
              handleBack();
            }}
            className="frosted-pill flex items-center gap-3 hover:scale-102 transition-transform cursor-pointer group"
          >
            <DataSurfLogo size={28} className="text-ocean-main" />
            {!selectedSpot && (
              <div className="flex flex-col">
                <span className="font-serif text-base font-bold leading-none tracking-tight text-ocean-deep">DataSurf</span>
                <span className="technical-label !text-[7px] opacity-40">California surf &amp; marine conditions</span>
              </div>
            )}
          </div>
        </div>

        <div className="pointer-events-auto flex flex-wrap items-center justify-end gap-2 sm:gap-4">
          {view === 'map' && !selectedSpot && (
            <button
              type="button"
              onClick={openAbout}
              className="frosted-pill flex items-center gap-2 px-3 py-2 text-[10px] font-bold uppercase tracking-tight text-ocean-deep/70 hover:bg-white/80"
            >
              <Info size={16} aria-hidden />
              About
            </button>
          )}

          <div className="flex items-center frosted-pill p-1 gap-0.5 sm:gap-1">
            {(
              [
                { id: 'surfer' as const, icon: Waves },
                { id: 'swimmer' as const, icon: Droplets },
                { id: 'family' as const, icon: Users },
              ] as const
            ).map(({ id, icon: Icon }) => (
              <button
                key={id}
                type="button"
                onClick={() => setUserProfile(id)}
                className={`flex items-center gap-1 px-2 py-1 sm:px-4 sm:py-1.5 rounded-full text-[8px] sm:text-[9px] font-bold uppercase transition-all ${
                  userProfile === id ? 'bg-ocean-main text-white' : 'text-ocean-deep/40 hover:bg-ocean-main/5'
                }`}
              >
                <Icon size={12} className="shrink-0 opacity-80" aria-hidden />
                <span className="hidden sm:inline">{id}</span>
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={() => setThemeMode((m) => (m === 'system' ? 'light' : m === 'light' ? 'dark' : 'system'))}
            className="frosted-pill p-2.5 text-ocean-deep hover:bg-white/80 transition-colors"
            title="Theme: system → light → dark"
            aria-label="Toggle color theme"
          >
            {themeMode === 'dark' ? (
              <Sun size={18} />
            ) : themeMode === 'light' ? (
              <Moon size={18} />
            ) : (
              <Sun size={18} className="opacity-60" />
            )}
          </button>
        </div>
      </header>

      <div
        className={`absolute inset-0 z-0 overflow-hidden transition-opacity duration-500 ${
          mapDimmed ? 'opacity-20 pointer-events-none' : 'opacity-100'
        }`}
      >
        <div className="absolute inset-0 z-10 pointer-events-none bg-gradient-to-br from-ocean-bright/5 via-transparent to-ocean-main/5 mix-blend-overlay" />
        <CoastalMap spots={spots} onSpotSelect={setSelectedSpot} />
      </div>

      <AnimatePresence mode="wait">
        {view === 'about' && <AboutPage key="about" onBack={closeAbout} />}
      </AnimatePresence>

      <AnimatePresence>
        {view === 'search' && !selectedSpot && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="fixed inset-0 z-40 bg-sand/95 backdrop-blur-xl overflow-y-auto px-6 pt-32 pb-32"
          >
            <div className="max-w-2xl mx-auto space-y-6">
              <div className="flex items-center justify-between">
                <h1 className="text-4xl">Spots</h1>
                <span className="technical-label">
                  {filteredSpots.length} of {spots.length}
                </span>
              </div>

              <div className="relative group">
                <Search
                  className="absolute left-6 top-1/2 -translate-y-1/2 text-ocean-deep/30 group-focus-within:text-ocean-main transition-colors"
                  size={20}
                  aria-hidden
                />
                <input
                  type="search"
                  placeholder="Search by name or county..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full h-16 pl-14 pr-6 bg-white rounded-3xl border-none shadow-xl shadow-ocean-main/5 text-ocean-deep placeholder:text-ocean-deep/20 focus:ring-2 focus:ring-ocean-main/20 transition-all outline-none"
                />
              </div>

              <div className="flex flex-col gap-1 max-h-[min(70vh,600px)] overflow-y-auto pr-1">
                {filteredSpots.map((spot) => (
                  <SpotListRow key={spot.id} spot={spot} onSelect={setSelectedSpot} favorites={favorites} />
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {view === 'saved' && !selectedSpot && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="fixed inset-0 z-40 bg-sand/95 backdrop-blur-xl overflow-y-auto px-6 pt-32 pb-32"
          >
            <div className="max-w-2xl mx-auto space-y-6">
              <div className="flex items-center justify-between">
                <h1 className="text-4xl">Saved</h1>
                <span className="technical-label">{savedSpots.length} pinned</span>
              </div>
              {savedSpots.length === 0 ? (
                <p className="font-serif italic text-lg text-ocean-deep/50 text-center py-16">
                  Star a spot from the detail view — it shows up here.
                </p>
              ) : (
                <div className="flex flex-col gap-1">
                  {savedSpots.map((spot) => (
                    <SpotListRow key={spot.id} spot={spot} onSelect={setSelectedSpot} favorites={favorites} />
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {!selectedSpot && view === 'map' && (
        <motion.div
          initial={{ y: '100%' }}
          animate={{ y: '0%' }}
          className="fixed bottom-0 left-0 right-0 z-30 max-w-lg mx-auto"
        >
          <div className="bg-white rounded-t-3xl shadow-2xl border-t border-slate-dark/5 overflow-hidden dark:bg-[#0f2438] dark:border-white/10">
            <div className="flex justify-center p-3">
              <div className="w-12 h-1 bg-slate-200 rounded-full dark:bg-white/20" />
            </div>

            <div className="px-6 pb-2 flex justify-between items-center">
              <h2 className="text-xl">Coastal spots</h2>
              <button
                type="button"
                onClick={() => {
                  setView('search');
                  setSelectedSpot(null);
                }}
                className="text-ocean-main text-sm font-semibold hover:underline"
              >
                See all {spots.length}
              </button>
            </div>

            <div className="px-6 pb-8 max-h-[40vh] overflow-y-auto space-y-1 pt-2">
              {bottomSheetSpots.map((spot) => (
                <SpotListRow key={spot.id} spot={spot} onSelect={setSelectedSpot} favorites={favorites} />
              ))}
            </div>
          </div>
        </motion.div>
      )}

      <AnimatePresence>
        {selectedSpot && view !== 'about' && (
          <motion.div
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed inset-0 z-40 bg-sand overflow-y-auto"
          >
            <div className="relative min-h-[40vh] w-full overflow-hidden bg-gradient-to-br from-ocean-main via-ocean-bright/90 to-seafoam">
              <div className="absolute inset-0 opacity-30 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-white/40 via-transparent to-transparent" />
              <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-sand to-transparent" />

              <motion.button
                whileTap={{ scale: 0.9 }}
                type="button"
                onClick={handleBack}
                className="absolute top-6 left-6 p-4 bg-white/90 backdrop-blur-xl rounded-2xl shadow-xl border border-white dark:bg-[#0f2438]/90 dark:border-white/10 z-10"
              >
                <ChevronLeft size={24} className="text-ocean-deep" />
              </motion.button>

              <div className="absolute top-6 right-6 flex gap-3 z-10">
                <motion.button
                  whileTap={{ scale: 0.9 }}
                  type="button"
                  onClick={handleShare}
                  className="p-4 bg-white/90 backdrop-blur-xl rounded-2xl shadow-xl border border-white dark:bg-[#0f2438]/90 dark:border-white/10"
                >
                  <Share2 size={20} className="text-ocean-deep" />
                </motion.button>
              </div>

              <div className="absolute bottom-16 left-8 md:left-12 right-8 md:right-12 z-10">
                <p className="text-white/90 font-mono text-[10px] uppercase tracking-[0.2em] mb-2">{selectedSpot.county}</p>
                <h2 className="text-4xl sm:text-5xl text-white font-serif tracking-tight leading-none drop-shadow-sm">
                  {selectedSpot.name}
                </h2>
                <p className="text-white/80 text-xs font-mono mt-3">
                  {selectedSpot.lat.toFixed(4)}°N · {Math.abs(selectedSpot.lng).toFixed(4)}°W
                </p>
              </div>
            </div>

            <div className="px-4 sm:px-8 md:px-12 py-6 max-w-7xl mx-auto space-y-6 pb-36">
              <div className="glass-panel rounded-2xl p-3 flex gap-3 border-ocean-main/15 bg-ocean-main/5" role="status">
                <Info size={14} className="text-ocean-main flex-shrink-0 mt-0.5" aria-hidden />
                <p className="text-xs text-ocean-deep/90 leading-relaxed">
                  Live marine and weather data from{' '}
                  <a href={OPEN_METEO} target="_blank" rel="noopener noreferrer" className="font-bold text-ocean-main underline underline-offset-2">
                    Open-Meteo
                  </a>
                  . Not a surf forecast service — check local reports and safety before paddling out.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
                <div className="space-y-8">
                  <div className="glass-panel rounded-2xl p-4 space-y-2">
                    <p className="technical-label">{profileScoreLabel}</p>
                    {loadingWeather ? (
                      <div className="h-8 w-32 rounded bg-ocean-main/10 animate-pulse" />
                    ) : surfScoreRounded != null ? (
                      <div className="flex items-center gap-3">
                        <span className="font-mono text-2xl font-bold text-ocean-deep">{surfScoreRounded}/5</span>
                        <span className="font-mono text-ocean-main text-lg" aria-hidden>
                          {Array.from({ length: 5 }, (_, i) => (
                            <span key={i} className={i < surfScoreRounded ? 'opacity-100' : 'opacity-20'}>
                              ●
                            </span>
                          ))}
                        </span>
                      </div>
                    ) : (
                      <p className="text-sm text-ocean-deep/50">Loading conditions…</p>
                    )}
                  </div>

                  <button
                    type="button"
                    onClick={() => toggleFavorite(selectedSpot.id)}
                    className={`w-full py-5 rounded-3xl font-bold tracking-tight transition-all shadow-sm border ${
                      favorites.includes(selectedSpot.id)
                        ? 'bg-ocean-main/15 text-ocean-main border-ocean-main/30'
                        : 'bg-white text-ocean-main hover:bg-ocean-main hover:text-white border-ocean-main/20 dark:bg-[#0f2438]'
                    }`}
                  >
                    {favorites.includes(selectedSpot.id) ? 'Saved ✓' : 'Save spot'}
                  </button>
                </div>

                <div className="space-y-10">
                  <section>
                    <h4 className="technical-label mb-4">7-day outlook (Open-Meteo)</h4>
                    {weatherError && (
                      <p className="text-xs text-risk-caution mb-2">Forecast unavailable.</p>
                    )}
                    <div className="space-y-2">
                      {(weather?.week ?? []).map((d) => (
                        <div
                          key={d.date}
                          className="glass-panel p-4 rounded-2xl flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-white/40 dark:hover:bg-[#0f2438]"
                        >
                          <div>
                            <span className="text-xs font-bold text-ocean-deep/50">{formatDayLabel(d.date)}</span>
                            <p className="text-sm font-serif font-bold text-ocean-deep mt-0.5">{d.weatherLabel}</p>
                          </div>
                          <div className="flex flex-wrap gap-4 text-[11px] font-mono text-ocean-deep/80">
                            <span>
                              Waves max{' '}
                              <strong className="text-ocean-deep">
                                {d.waveHeightMaxFt != null ? `${d.waveHeightMaxFt} ft` : '—'}
                              </strong>
                            </span>
                            <span>
                              Swell max{' '}
                              <strong className="text-ocean-deep">
                                {d.swellHeightMaxFt != null ? `${d.swellHeightMaxFt} ft` : '—'}
                              </strong>
                              {d.swellPeriodMaxS != null ? ` @ ${d.swellPeriodMaxS}s` : ''}
                            </span>
                            <span>
                              Air max <strong>{d.tempMaxF != null ? `${d.tempMaxF}°F` : '—'}</strong>
                            </span>
                            <span>
                              Wind max <strong>{d.windMaxMph != null ? `${d.windMaxMph} mph` : '—'}</strong>
                            </span>
                            <span>Rain {d.precipProbability}%</span>
                            <span>UV {d.uvIndexMax ?? '—'}</span>
                          </div>
                        </div>
                      ))}
                      {!loadingWeather && (!weather?.week || weather.week.length === 0) && (
                        <p className="text-sm text-ocean-deep/50">No multi-day data.</p>
                      )}
                    </div>
                  </section>

                  <section>
                    <h4 className="technical-label mb-4">Right now</h4>
                    {weatherError && (
                      <div className="mb-4 flex items-start gap-2 text-xs text-risk-caution bg-risk-caution/10 rounded-xl px-3 py-2 border border-risk-caution/20">
                        <Info size={14} className="shrink-0 mt-0.5" aria-hidden />
                        <span>Could not load live conditions. Try again in a moment.</span>
                      </div>
                    )}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                      {loadingWeather
                        ? Array.from({ length: 4 }).map((_, i) => (
                            <div key={i} className="glass-panel p-5 rounded-[32px] animate-pulse space-y-3">
                              <div className="h-4 w-4 rounded bg-ocean-main/15" />
                              <div className="h-4 w-20 rounded bg-ocean-main/10" />
                              <div className="h-6 w-16 rounded bg-ocean-main/10" />
                            </div>
                          ))
                        : [
                            {
                              label: 'Waves',
                              val: weather?.waveHeight != null ? `${weather.waveHeight.toFixed(1)} ft` : '—',
                              sub: weather?.condition ?? undefined,
                              icon: Waves,
                            },
                            {
                              label: 'Wind',
                              val: weather?.windSpeed != null ? `${weather.windSpeed} mph` : '—',
                              sub:
                                weather &&
                                `${formatWindRelationLabel(weather.onshoreOffshore)}${
                                  weather.windGusts != null ? ` · Gusts ${weather.windGusts} mph` : ''
                                }`,
                              icon: WindIcon,
                            },
                            {
                              label: 'Water temp',
                              val: weather?.waterTemp != null ? `${weather.waterTemp}°F` : '—',
                              sub:
                                weather?.feelsLike != null ? `Air feels like ${weather.feelsLike}°F` : undefined,
                              icon: Thermometer,
                            },
                            {
                              label: 'Air quality',
                              val: weather?.aqi != null ? `${weather.aqi} US AQI` : '—',
                              sub: 'Open-Meteo air quality',
                              icon: Activity,
                            },
                          ].map((item, i) => (
                            <div key={i} className="glass-panel p-5 rounded-[32px] flex flex-col gap-4 border-white/60">
                              <item.icon size={18} className="text-ocean-main" />
                              <div className="space-y-0.5">
                                <span className="technical-label !text-[8px] opacity-40">{item.label}</span>
                                <p className="text-base font-bold text-ocean-deep leading-none">{item.val}</p>
                                {item.sub && (
                                  <p className="text-[9px] font-mono text-ocean-main/70 mt-0.5 leading-snug">{item.sub}</p>
                                )}
                              </div>
                            </div>
                          ))}
                    </div>
                  </section>

                  <section>
                    <h4 className="technical-label mb-4">Swell &amp; extras</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {[
                        {
                          label: 'Primary swell',
                          val:
                            weather?.swellHeight != null && weather.swellHeight > 0 && weather.swellPeriod > 0
                              ? `${weather.swellHeight.toFixed(1)} ft @ ${Math.round(weather.swellPeriod)}s`
                              : weather?.waveHeight != null && weather.wavePeriod > 0
                                ? `${weather.waveHeight.toFixed(1)} ft @ ${Math.round(weather.wavePeriod)}s`
                                : '—',
                          sub:
                            weather?.swellDirection != null && weather.swellHeight > 0
                              ? `From ${weather.swellDirection}°`
                              : weather?.waveDirection != null
                                ? `Sea from ${weather.waveDirection}°`
                                : undefined,
                          icon: Waves,
                        },
                        {
                          label: 'UV index',
                          val: weather?.uvIndex != null ? `${weather.uvIndex}` : '—',
                          sub: weather?.uvIndex != null ? uvIndexLabel(weather.uvIndex) : undefined,
                          icon: Sun,
                        },
                        {
                          label: 'Rain chance',
                          val: weather?.precipProbability != null ? `${weather.precipProbability}%` : '—',
                          sub: 'Daily max (today)',
                          icon: Droplets,
                        },
                        {
                          label: 'Sunset',
                          val: weather?.sunset ?? '—',
                          sub: weather?.sunrise ? `Sunrise ${weather.sunrise}` : undefined,
                          icon: Sun,
                        },
                      ].map((item, i) => (
                        <div key={i} className="glass-panel p-5 rounded-[32px] border-white/30">
                          <item.icon size={16} className="text-ocean-main/60 mb-3" />
                          <p className="technical-label !text-[8px] opacity-40 mb-1">{item.label}</p>
                          <p className="text-sm font-bold text-ocean-deep">{item.val}</p>
                          {item.sub && (
                            <p className="text-[9px] font-mono text-ocean-main/60 capitalize mt-0.5">{item.sub}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </section>

                  <div className="pt-8 border-t border-ocean-main/5 flex flex-col md:flex-row justify-between items-center gap-4 opacity-50 text-center md:text-left">
                    <p className="text-[10px] font-mono max-w-md">
                      Weather &amp; marine: Open-Meteo. Spot list from project station metadata.
                    </p>
                    <button type="button" onClick={openAbout} className="technical-label hover:text-ocean-main">
                      About &amp; attribution
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {view !== 'about' && (
        <nav className="fixed bottom-8 left-1/2 -translate-x-1/2 frosted-pill px-6 sm:px-8 py-4 z-50 flex items-center gap-8 sm:gap-12 transition-all hover:bg-white shadow-2xl max-w-[95vw]">
          <motion.button
            whileTap={{ scale: 0.9 }}
            type="button"
            onClick={() => {
              setView('map');
              setSelectedSpot(null);
            }}
            className={`flex flex-col items-center gap-1 transition-all ${view === 'map' ? 'text-ocean-main scale-110' : 'text-ocean-deep/30 hover:text-ocean-deep'}`}
          >
            <MapIcon size={20} />
            <span className="text-[8px] font-sans font-bold uppercase tracking-widest">Map</span>
          </motion.button>

          <div className="h-6 w-px bg-ocean-main/10" />

          <motion.button
            whileTap={{ scale: 0.9 }}
            type="button"
            onClick={() => {
              setView('search');
              setSelectedSpot(null);
            }}
            className={`flex flex-col items-center gap-1 transition-all ${view === 'search' ? 'text-ocean-main scale-110' : 'text-ocean-deep/30 hover:text-ocean-deep'}`}
          >
            <Search size={20} />
            <span className="text-[8px] font-sans font-bold uppercase tracking-widest">Search</span>
          </motion.button>

          <div className="h-6 w-px bg-ocean-main/10" />

          <motion.button
            whileTap={{ scale: 0.9 }}
            type="button"
            onClick={() => {
              setView('saved');
              setSelectedSpot(null);
            }}
            className={`flex flex-col items-center gap-1 transition-all ${view === 'saved' ? 'text-ocean-main scale-110' : 'text-ocean-deep/30 hover:text-ocean-deep'}`}
          >
            <Star size={20} />
            <span className="text-[8px] font-sans font-bold uppercase tracking-widest">Saved</span>
          </motion.button>
        </nav>
      )}
    </div>
  );
}
