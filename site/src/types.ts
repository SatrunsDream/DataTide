/** California surf spot (from `public/data/surf_spots.json`). */
export interface SurfSpot {
  id: string;
  name: string;
  county: string;
  lat: number;
  lng: number;
  /** Degrees; used for onshore/offshore wind heuristic (default ~270 = west-facing). */
  shorelineBearing?: number;
}
