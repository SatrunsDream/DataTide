import React from 'react';
import { motion } from 'motion/react';

export const DataSurfLogo = ({ size = 32, className = "" }: { size?: number, className?: string }) => {
  return (
    <svg 
      width={size} 
      height={size} 
      viewBox="0 0 100 100" 
      fill="none" 
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Editorial Mark: Geometric Wave within a circle */}
      <circle cx="50" cy="50" r="48" stroke="currentColor" strokeWidth="1" strokeOpacity="0.2" />
      <motion.path 
        d="M20 50 C 20 50, 35 30, 50 50 C 65 70, 80 50, 80 50" 
        stroke="currentColor" 
        strokeWidth="6" 
        strokeLinecap="round" 
        strokeLinejoin="round" 
        animate={{ 
          y: [0, -4, 0],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      <motion.path 
        d="M25 65 C 25 65, 38 52, 50 65 C 62 78, 75 65, 75 65" 
        stroke="currentColor" 
        strokeWidth="3" 
        strokeLinecap="round" 
        strokeOpacity="0.4"
        animate={{ 
          y: [0, 4, 0],
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 0.5
        }}
      />
    </svg>
  );
};

export const ProbabilityCurve = ({ intensity, color }: { intensity: number, color: string }) => {
    // Generate a bell curve path
    const points = [];
    const width = 120;
    const height = 40;
    const mean = width / 2;
    const sigma = 15;

    for (let x = 0; x <= width; x += 2) {
        const exponent = -Math.pow(x - mean, 2) / (2 * Math.pow(sigma, 2));
        const y = height - (Math.exp(exponent) * height * intensity);
        points.push(`${x},${y}`);
    }

    const pathData = `M 0,${height} L ${points.join(' L ')} L ${width},${height} Z`;

    return (
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="curve-mask">
            <path d={pathData} fill={color} fillOpacity="0.2" stroke={color} strokeWidth="1" />
            
            {/* The "Most Likely" indicator */}
            <line 
                x1={mean} 
                y1={height} 
                x2={mean} 
                y2={height - (height * intensity)} 
                stroke={color} 
                strokeWidth="1" 
                strokeDasharray="2,2" 
            />
        </svg>
    );
};
