import React, { useState } from 'react';
import { motion } from 'motion/react';
import Map, { NavigationControl, MapLayerMouseEvent, Marker } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { SurfSpot } from '../types';

const MARKER = '#0d5c8c';

interface CoastalMapProps {
  spots: SurfSpot[];
  onSpotSelect: (spot: SurfSpot | null) => void;
}

export const CoastalMap = ({ spots, onSpotSelect }: CoastalMapProps) => {
  const [viewState, setViewState] = useState({
    longitude: -119.2,
    latitude: 35.8,
    zoom: 6,
    pitch: 0,
    bearing: 0,
  });

  const onMapClick = (_event: MapLayerMouseEvent) => {
    onSpotSelect(null);
  };

  return (
    <div className="w-full h-full relative font-sans">
      <Map
        {...viewState}
        onMove={(evt) => setViewState(evt.viewState)}
        mapStyle="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
        onClick={onMapClick}
      >
        {spots.map((spot) => (
          <Marker
            key={spot.id}
            longitude={spot.lng}
            latitude={spot.lat}
            anchor="bottom"
            onClick={(e) => {
              e.originalEvent.stopPropagation();
              onSpotSelect(spot);
            }}
          >
            <motion.div whileHover={{ scale: 1.15 }} className="cursor-pointer relative group">
              <div
                className="w-3.5 h-3.5 rounded-full border-2 border-white shadow-lg"
                style={{ backgroundColor: MARKER }}
              />
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                <div className="glass-panel px-3 py-1.5 rounded-xl whitespace-nowrap">
                  <span className="text-[9px] font-bold text-ocean-deep leading-none">{spot.name}</span>
                  <span className="block text-[8px] text-ocean-deep/50 mt-0.5">{spot.county}</span>
                </div>
              </div>
            </motion.div>
          </Marker>
        ))}
        <NavigationControl position="bottom-right" />
      </Map>

      <div className="absolute bottom-10 right-10 z-10 hidden md:block pointer-events-none">
        <div className="frosted-pill flex items-center gap-4 text-[10px] font-mono font-bold tracking-tight">
          <span>{viewState.latitude.toFixed(2)}°N</span>
          <span>{Math.abs(viewState.longitude).toFixed(2)}°W</span>
        </div>
      </div>
    </div>
  );
};
