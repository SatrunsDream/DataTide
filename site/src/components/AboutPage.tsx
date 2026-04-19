import React from 'react';
import { motion } from 'motion/react';
import { ChevronLeft } from 'lucide-react';

type AboutPageProps = {
  onBack: () => void;
};

export function AboutPage({ onBack }: AboutPageProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 8 }}
      className="fixed inset-0 z-[60] bg-sand overflow-y-auto px-6 pt-28 pb-32"
    >
      <div className="max-w-2xl mx-auto space-y-10">
        <button
          type="button"
          onClick={onBack}
          className="frosted-pill flex items-center gap-2 text-ocean-deep hover:bg-white transition-colors"
        >
          <ChevronLeft size={20} />
          <span className="text-sm font-semibold">Back</span>
        </button>

        <div>
          <h1 className="text-4xl mb-4">About DataSurf</h1>
          <p className="text-ocean-deep/80 leading-relaxed">
            DataSurf is a California surf conditions interface. Pick a coastal spot, then read live marine and
            weather data from Open-Meteo: waves, swell, wind, water temperature, UV, rain chance, and a seven-day
            outlook—without relying on stock beach photos or water-bacteria modeling in the UI.
          </p>
          <p className="text-ocean-deep/80 leading-relaxed mt-4">
            The simple score blends wave and wind heuristics for surfers, swimmers, or families; it is not a
            professional surf forecast.
          </p>
        </div>

        <section className="space-y-4">
          <h2 className="text-2xl font-serif">What you are seeing</h2>
          <ul className="list-disc pl-5 space-y-2 text-ocean-deep/80 text-sm leading-relaxed">
            <li>
              <strong className="text-ocean-deep">Open-Meteo</strong> provides forecast weather, marine parameters,
              and air quality for the spot coordinates (no API key).
            </li>
            <li>
              <strong className="text-ocean-deep">Spot list</strong> comes from project station metadata (name,
              county, latitude/longitude, shoreline bearing when available).
            </li>
            <li>
              <strong className="text-ocean-deep">Wind vs shore</strong> uses shoreline bearing to label onshore,
              offshore, or cross-shore conditions—useful but approximate.
            </li>
          </ul>
        </section>

        <section className="space-y-4">
          <h2 className="text-2xl font-serif">Data sources</h2>
          <ul className="space-y-2 text-sm">
            <li>
              <a
                href="https://open-meteo.com/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-ocean-main font-semibold hover:underline"
              >
                Open-Meteo — weather, marine, and air quality APIs (this app)
              </a>
            </li>
            <li>
              <a href="https://cdip.ucsd.edu/" target="_blank" rel="noopener noreferrer" className="text-ocean-main font-semibold hover:underline">
                CDIP — wave buoys (reference; not wired into this build)
              </a>
            </li>
          </ul>
        </section>

        <div className="glass-panel rounded-2xl p-5 border-ocean-main/20 bg-ocean-main/5">
          <p className="text-sm text-ocean-deep leading-relaxed">
            <strong>Disclaimer:</strong> Conditions change quickly. Check swell models, tides, local knowledge, and
            safety before entering the water. This app does not replace lifeguards or official notices.
          </p>
        </div>

        <section className="space-y-4">
          <h2 className="text-2xl font-serif">County health &amp; beach info</h2>
          <ul className="space-y-2 text-sm">
            <li>
              <a
                href="http://publichealth.lacounty.gov/beach/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-ocean-main font-semibold hover:underline"
              >
                Los Angeles County — Beach water quality
              </a>
            </li>
            <li>
              <a
                href="https://www.sdbeachinfo.com/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-ocean-main font-semibold hover:underline"
              >
                San Diego County — Beach information
              </a>
            </li>
            <li>
              <a
                href="https://www.santamonica.gov/beach"
                target="_blank"
                rel="noopener noreferrer"
                className="text-ocean-main font-semibold hover:underline"
              >
                City of Santa Monica — Beach &amp; ocean
              </a>
            </li>
          </ul>
        </section>
      </div>
    </motion.div>
  );
}
