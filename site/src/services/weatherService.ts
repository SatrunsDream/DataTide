/**
 * Open-Meteo — weather, marine, air quality, and 7-day outlook (no API key).
 * https://open-meteo.com/
 */

const CACHE_TTL_MS = 15 * 60 * 1000;
const FETCH_TIMEOUT_MS = 10_000;

const cache = new Map<string, { data: SurfConditions; timestamp: number }>();

export interface DayOutlook {
  date: string;
  waveHeightMaxFt: number | null;
  swellHeightMaxFt: number | null;
  swellPeriodMaxS: number | null;
  tempMaxF: number | null;
  windMaxMph: number | null;
  precipProbability: number;
  uvIndexMax: number | null;
  weatherLabel: string;
}

export interface WeatherData {
  temperature: number;
  feelsLike: number;
  windSpeed: number;
  windGusts: number;
  windDirection: number;
  uvIndex: number;
  condition: string;
  precipProbability: number;
  waveHeight: number;
  wavePeriod: number;
  waveDirection: number;
  swellHeight: number;
  swellPeriod: number;
  swellDirection: number;
  waterTemp: number;
  aqi: number;
  sunrise: string;
  sunset: string;
  onshoreOffshore: 'onshore' | 'offshore' | 'cross-shore' | 'glassy';
}

export interface SurfConditions extends WeatherData {
  /** Next 7 days (marine + weather daily). */
  week: DayOutlook[];
}

export function uvIndexLabel(uv: number): string {
  if (uv <= 2) return 'Low';
  if (uv <= 5) return 'Moderate';
  if (uv <= 7) return 'High';
  if (uv <= 10) return 'Very high';
  return 'Extreme';
}

export function formatWindRelationLabel(
  r: 'onshore' | 'offshore' | 'cross-shore' | 'glassy',
): string {
  const m: Record<typeof r, string> = {
    onshore: 'Onshore',
    offshore: 'Offshore',
    'cross-shore': 'Cross-shore',
    glassy: 'Light / glassy',
  };
  return m[r];
}

function mapWeatherCode(code: number): string {
  if (code === 0) return 'Clear';
  if (code === 1) return 'Mainly clear';
  if (code === 2) return 'Partly cloudy';
  if (code === 3) return 'Overcast';
  if (code === 45 || code === 48) return 'Foggy';
  if (code >= 51 && code <= 55) return 'Drizzle';
  if (code >= 61 && code <= 65) return 'Rain';
  if (code >= 71 && code <= 77) return 'Snow';
  if (code >= 80 && code <= 82) return 'Showers';
  if (code >= 95 && code <= 99) return 'Thunderstorm';
  return 'Cloudy';
}

async function fetchJsonWithTimeout(url: string): Promise<unknown> {
  const ctrl = new AbortController();
  const timeout = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    const r = await fetch(url, { signal: ctrl.signal });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } finally {
    clearTimeout(timeout);
  }
}

export const calculateWindRelation = (
  windDirection: number,
  shorelineBearing: number,
): 'onshore' | 'offshore' | 'cross-shore' | 'glassy' => {
  if (windDirection == null) return 'glassy';
  const diff = Math.abs((windDirection - shorelineBearing + 360) % 360);
  if (diff < 45 || diff > 315) return 'onshore';
  if (diff > 135 && diff < 225) return 'offshore';
  return 'cross-shore';
};

function normalizeSeaSurfaceTempF(raw: number | undefined, fallback = 64): number {
  if (raw == null || Number.isNaN(raw)) return fallback;
  if (raw >= -2 && raw < 40) {
    return Math.round((raw * 9) / 5 + 32);
  }
  return Math.round(raw);
}

function parseWeek(
  weatherRes: Record<string, unknown>,
  marineRes: Record<string, unknown>,
): DayOutlook[] {
  const dW = (weatherRes?.daily as Record<string, unknown[]>) ?? {};
  const dM = (marineRes?.daily as Record<string, unknown[]>) ?? {};
  const times = (dW.time as string[] | undefined) ?? [];
  const out: DayOutlook[] = [];
  for (let i = 0; i < times.length; i++) {
    const date = times[i].slice(0, 10);
    const wc = (dW.weather_code as number[] | undefined)?.[i];
    const wh = (dM.wave_height_max as number[] | undefined)?.[i];
    const sh = (dM.swell_wave_height_max as number[] | undefined)?.[i];
    const sp = (dM.wave_period_max as number[] | undefined)?.[i];
    const tmax = (dW.temperature_2m_max as number[] | undefined)?.[i];
    const wmax = (dW.wind_speed_10m_max as number[] | undefined)?.[i];
    const pcp = (dW.precipitation_probability_max as number[] | undefined)?.[i];
    const uvi = (dW.uv_index_max as number[] | undefined)?.[i];
    out.push({
      date,
      waveHeightMaxFt: wh != null ? parseFloat(Number(wh).toFixed(1)) : null,
      swellHeightMaxFt: sh != null ? parseFloat(Number(sh).toFixed(1)) : null,
      swellPeriodMaxS: sp != null ? parseFloat(Number(sp).toFixed(1)) : null,
      tempMaxF: tmax != null ? Math.round(Number(tmax)) : null,
      windMaxMph: wmax != null ? Math.round(Number(wmax)) : null,
      precipProbability: pcp != null ? Math.round(Number(pcp)) : 0,
      uvIndexMax: uvi != null ? Math.round(Number(uvi)) : null,
      weatherLabel: wc != null ? mapWeatherCode(Number(wc)) : '—',
    });
  }
  return out;
}

function buildSurfConditions(
  weatherRes: Record<string, unknown>,
  marineRes: Record<string, unknown>,
  airRes: Record<string, unknown>,
  shorelineBearing: number,
): SurfConditions {
  const cW = (weatherRes?.current as Record<string, unknown>) ?? {};
  const dW = (weatherRes?.daily as Record<string, unknown[]>) ?? {};
  const cM = (marineRes?.current as Record<string, unknown>) ?? {};
  const cA = (airRes?.current as Record<string, unknown>) ?? {};

  const windDir = (cW.wind_direction_10m as number) ?? 270;
  const uvIndex = Math.round((dW.uv_index_max?.[0] as number) ?? 5);
  const precipProb = Math.round((dW.precipitation_probability_max?.[0] as number) ?? 0);
  const sstRaw = cM.sea_surface_temperature as number | undefined;
  const waterTemp = normalizeSeaSurfaceTempF(sstRaw);

  const sunriseRaw = (dW.sunrise?.[0] as string | undefined) ?? 'T06:00';
  const sunsetRaw = (dW.sunset?.[0] as string | undefined) ?? 'T19:00';
  const sunrise = sunriseRaw.includes('T') ? sunriseRaw.split('T')[1].slice(0, 5) : sunriseRaw.slice(0, 5);
  const sunset = sunsetRaw.includes('T') ? sunsetRaw.split('T')[1].slice(0, 5) : sunsetRaw.slice(0, 5);

  const base: WeatherData = {
    temperature: Math.round((cW.temperature_2m as number) ?? 72),
    feelsLike: Math.round((cW.apparent_temperature as number) ?? 72),
    windSpeed: Math.round((cW.wind_speed_10m as number) ?? 8),
    windGusts: Math.round((cW.wind_gusts_10m as number) ?? 12),
    windDirection: windDir,
    uvIndex,
    condition: mapWeatherCode((cW.weather_code as number) ?? 0),
    precipProbability: precipProb,
    waveHeight: parseFloat(((cM.wave_height as number) ?? 3.2).toFixed(1)),
    wavePeriod: parseFloat(((cM.wave_period as number) ?? 11).toFixed(1)),
    waveDirection: Math.round((cM.wave_direction as number) ?? 270),
    swellHeight: parseFloat(((cM.swell_wave_height as number) ?? 2.5).toFixed(1)),
    swellPeriod: parseFloat(((cM.swell_wave_period as number) ?? 12).toFixed(1)),
    swellDirection: Math.round((cM.swell_wave_direction as number) ?? 270),
    waterTemp,
    aqi: Math.round((cA.us_aqi as number) ?? 20),
    sunrise,
    sunset,
    onshoreOffshore: calculateWindRelation(windDir, shorelineBearing),
  };

  return {
    ...base,
    week: parseWeek(weatherRes, marineRes),
  };
}

export async function fetchBeachWeather(
  lat: number,
  lng: number,
  shorelineBearing: number = 270,
): Promise<SurfConditions> {
  const key = `${lat.toFixed(4)},${lng.toFixed(4)}`;
  const hit = cache.get(key);
  if (hit && Date.now() - hit.timestamp < CACHE_TTL_MS) {
    return hit.data;
  }

  const weatherUrl =
    `https://api.open-meteo.com/v1/forecast` +
    `?latitude=${lat}&longitude=${lng}` +
    `&current=temperature_2m,apparent_temperature,wind_speed_10m,wind_direction_10m,wind_gusts_10m,weather_code` +
    `&daily=sunrise,sunset,uv_index_max,precipitation_probability_max,weather_code,temperature_2m_max,wind_speed_10m_max` +
    `&temperature_unit=fahrenheit&wind_speed_unit=mph` +
    `&timezone=auto&forecast_days=7`;

  const marineUrl =
    `https://marine-api.open-meteo.com/v1/marine` +
    `?latitude=${lat}&longitude=${lng}` +
    `&current=wave_height,wave_period,wave_direction,swell_wave_height,swell_wave_period,swell_wave_direction,sea_surface_temperature` +
    `&daily=wave_height_max,swell_wave_height_max,wave_period_max` +
    `&length_unit=imperial&temperature_unit=fahrenheit` +
    `&timezone=auto&forecast_days=7`;

  const airUrl =
    `https://air-quality-api.open-meteo.com/v1/air-quality` +
    `?latitude=${lat}&longitude=${lng}&current=us_aqi`;

  const [weatherRes, marineRes, airRes] = await Promise.all([
    fetchJsonWithTimeout(weatherUrl),
    fetchJsonWithTimeout(marineUrl),
    fetchJsonWithTimeout(airUrl),
  ]);

  const data = buildSurfConditions(
    weatherRes as Record<string, unknown>,
    marineRes as Record<string, unknown>,
    airRes as Record<string, unknown>,
    shorelineBearing,
  );

  cache.set(key, { data, timestamp: Date.now() });
  return data;
}

/** Surf quality 1–5 from current marine + wind (Open-Meteo only). */
export function calculateSurfScore(
  w: SurfConditions,
  profile: 'surfer' | 'swimmer' | 'family',
): number {
  const period = w.swellHeight > 0.3 ? w.swellPeriod : w.wavePeriod;
  const height = w.swellHeight > 0.3 ? w.swellHeight : w.waveHeight;
  let surfScore = 3;
  if (height > 2 && height < 7) surfScore += 1;
  if (period > 10) surfScore += 1;
  if (height > 10 || height < 1) surfScore -= 2;
  let windScore = 3;
  if (w.onshoreOffshore === 'offshore') windScore = 5;
  if (w.onshoreOffshore === 'glassy') windScore = 4;
  if (w.onshoreOffshore === 'onshore') windScore = 1;
  const weights = {
    surfer: { surf: 0.65, wind: 0.35 },
    swimmer: { surf: 0.4, wind: 0.6 },
    family: { surf: 0.35, wind: 0.65 },
  }[profile];
  const combined = surfScore * weights.surf + windScore * weights.wind;
  return Math.min(5, Math.max(1, combined));
}
