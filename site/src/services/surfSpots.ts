import type { SurfSpot } from '../types';

export type SurfSpotsPayload = {
  source?: string;
  n?: number;
  spots: SurfSpot[];
};

const URL = '/data/surf_spots.json';

export async function loadSurfSpots(): Promise<SurfSpot[]> {
  try {
    const res = await fetch(URL, { cache: 'no-store' });
    if (!res.ok) return [];
    const data = (await res.json()) as SurfSpotsPayload;
    return Array.isArray(data.spots) ? data.spots : [];
  } catch {
    return [];
  }
}
